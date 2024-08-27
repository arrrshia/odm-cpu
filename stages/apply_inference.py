from opendm import types
from opendm import io
from opendm import system
from PIL import Image
from opendm import log
import requests
from opensfm.actions import undistort
from opensfm.dataset import DataSet
from opendm.osfm import OSFMContext
import os
from opendm import log
from opendm import io
import zipfile
from opendm import system
from opendm import context
from opendm import types
from opendm.multispectral import get_primary_band_name
from opendm.photo import find_largest_photo_dim
from opendm.objpacker import obj_pack
from opendm.gltf import obj2glb
import shutil
import time
from opensfm.undistort import add_image_format_extension

def read_task_id_from_file(file_path):
    """
    Reads the task ID from a given text file.

    Args:
    - file_path (str): The path to the text file containing the task ID.

    Returns:
    - str: The task ID read from the file, or None if the file cannot be read.
    """
    try:
        with open(file_path, 'r') as file:
            task_id = file.read().strip()  # Read the task ID and remove any leading/trailing whitespace
            return task_id
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

    return None

# Example usage:
# task_id = read_task_id_from_file('task_id.txt')
# print(task_id)

class ODMInferenceStage(types.ODM_Stage):
    def process(self, args, outputs):
        tree = outputs['tree']
        reconstruction = outputs['reconstruction']
        octx = OSFMContext(tree.opensfm)
        base_url="http://web:4000"
        # The task ID you want to check
        task_id = read_task_id_from_file(os.path.join(tree.opensfm, 'task_id.txt'))
        # The endpoint to check the task status
        status_url = f'{base_url}/check_task/{str(task_id)}'
        print(status_url)
        # Loop until the task is successful or fails
        while True:
            # Make a GET request to the Flask endpoint that checks the task status
            response = requests.get(status_url)

            # Check if the request was successful
            if response:
                status_code = response.json()
                status = status_code['status']

                # If the task was successful, break out of the loop
                if status == 'SUCCESS':
                    print('Task completed successfully!')
                    break
                elif status == 'FAILURE':
                    print('Task failed.')
                    break
                else:
                    print(f'Task still processing. Current status: {status}')
            else:
                print(f'Failed to retrieve task status, status code: {response.status_code}')

            # Wait for a short period before making another request to avoid overloading the server
            time.sleep(10)  # Sleep for 10 seconds

        url = 'http://web:4000/files/download'
        response = requests.get(url)

        for filename in os.listdir(tree.infer_image_outputdir):
            file_path = os.path.join(tree.infer_image_outputdir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

        if response.status_code == 200:
            with open(os.path.join(tree.infer_image_outputdir, 'images.zip'), 'wb') as f:
                f.write(response.content)
                print("Success")
        else:
            print("Failure")
        if not os.path.exists(tree.infer_image_outputdir):
            os.makedirs(tree.infer_image_outputdir)
        with zipfile.ZipFile(tree.infer_image_outputdir+"/images.zip",'r') as zip_ref:
            zip_ref.extractall(tree.infer_image_outputdir)
            print("Extracted all images")
        
        undistort_pipeline = []

        def undistort_callback(shot_id, image):
            for func in undistort_pipeline:
                image = func(shot_id, image)
            return image
        # Below undistorts
        inference_image_list = os.listdir(tree.infer_image_outputdir)
        inference_image_list_override = [os.path.join(tree.infer_image_outputdir, name) for name in inference_image_list]       
        octx.convert_and_undistort_inference_images(tree.infermod, self.rerun(), undistort_callback, inference_image_list_override)


        max_dim = find_largest_photo_dim(reconstruction.photos)
        max_texture_size = 8 * 1024 # default

        if max_dim > 8000:
            log.ODM_INFO("Large input images (%s pixels), increasing maximum texture size." % max_dim)
            max_texture_size *= 3

        class nonloc:
            runs = []

        # Create inference model output directories
        def inference_run(nvm_file, primary=True, band=None):
            subdir = ""
            if not primary and band is not None:
                subdir = band
                                       # Copy over nvm file from opensfm
            inference_undistorted = os.path.join(tree.infermod, "undistorted")
            shutil.copy(tree.opensfm_reconstruction_nvm, inference_undistorted)            
            nonloc.runs += [{
                'out_dir': tree.infer_3dmodel,
                'model': tree.odm_mesh,
                'nadir': False,
                'primary': primary,
                'nvm_file': os.path.join(tree.infermod, 'undistorted/reconstruction.nvm'), #os.path.join(tree.infermod, 'reconstruction.nvm')
                'labeling_file': None #os.path.join(tree.odm_texturing, "odm_textured_model_geo_labeling.vec") if subdir else None
            }]

        if reconstruction.multi_camera:

            for band in reconstruction.multi_camera:
                primary = band['name'] == get_primary_band_name(reconstruction.multi_camera, args.primary_band)
                nvm_file = os.path.join(tree.opensfm, "undistorted", "reconstruction_%s.nvm" % band['name'].lower())
                inference_run(nvm_file, primary, band['name'].lower())
            
            # Sort to make sure primary band is processed first
            nonloc.runs.sort(key=lambda r: r['primary'], reverse=True)
        else:
            inference_run(tree.opensfm_reconstruction_nvm)

        progress_per_run = 100.0 / len(nonloc.runs)
        progress = 0.0

        for r in nonloc.runs:
            if not io.dir_exists(r['out_dir']):
                system.mkdir_p(r['out_dir'])

            odm_textured_model_obj = os.path.join(r['out_dir'], tree.odm_textured_model_obj)
            unaligned_obj = io.related_file_path(odm_textured_model_obj, postfix="_unaligned")

            if not io.file_exists(odm_textured_model_obj) or self.rerun():
                log.ODM_INFO('Writing MVS Textured file in: %s'
                              % odm_textured_model_obj)

                if os.path.isfile(unaligned_obj):
                    os.unlink(unaligned_obj)

                # Format arguments to fit Mvs-Texturing app
                skipGlobalSeamLeveling = ""
                skipLocalSeamLeveling = ""
                keepUnseenFaces = ""
                nadir = ""

                if args.texturing_skip_global_seam_leveling:
                    skipGlobalSeamLeveling = "--skip_global_seam_leveling"
                if args.texturing_skip_local_seam_leveling:
                    skipLocalSeamLeveling = "--skip_local_seam_leveling"
                if args.texturing_keep_unseen_faces:
                    keepUnseenFaces = "--keep_unseen_faces"
                if (r['nadir']):
                    nadir = '--nadir_mode'

                # mvstex definitions
                kwargs = {
                    'bin': context.mvstex_path,
                    'out_dir': os.path.join(r['out_dir'], "odm_textured_model_geo"),
                    'model': r['model'],
                    'dataTerm': 'gmi',
                    'outlierRemovalType': 'gauss_clamping',
                    'skipGlobalSeamLeveling': skipGlobalSeamLeveling,
                    'skipLocalSeamLeveling': skipLocalSeamLeveling,
                    'keepUnseenFaces': keepUnseenFaces,
                    'toneMapping': 'none',
                    'nadirMode': nadir,
                    'maxTextureSize': '--max_texture_size=%s' % max_texture_size,
                    'nvm_file': r['nvm_file'],
                    'intermediate': '--no_intermediate_results' if (r['labeling_file'] or not reconstruction.multi_camera) else '',
                    'labelingFile': '-L "%s"' % r['labeling_file'] if r['labeling_file'] else ''
                }

                mvs_tmp_dir = os.path.join(r['out_dir'], 'tmp')

                # Make sure tmp directory is empty
                if io.dir_exists(mvs_tmp_dir):
                    log.ODM_INFO("Removing old tmp directory {}".format(mvs_tmp_dir))
                    shutil.rmtree(mvs_tmp_dir)

                # run texturing binary
                system.run('"{bin}" "{nvm_file}" "{model}" "{out_dir}" '
                        '-d {dataTerm} -o {outlierRemovalType} '
                        '-t {toneMapping} '
                        '{intermediate} '
                        '{skipGlobalSeamLeveling} '
                        '{skipLocalSeamLeveling} '
                        '{keepUnseenFaces} '
                        '{nadirMode} '
                        '{labelingFile} '
                        '{maxTextureSize} '.format(**kwargs))

                if r['primary'] and (not r['nadir'] or args.skip_3dmodel):
                    # GlTF?
                    if args.gltf:
                        log.ODM_INFO("Generating glTF Binary")
                        odm_textured_model_glb = os.path.join(r['out_dir'], tree.odm_textured_model_glb)
            
                        try:
                            obj2glb(odm_textured_model_obj, odm_textured_model_glb, rtc=reconstruction.get_proj_offset(), _info=log.ODM_INFO)
                        except Exception as e:
                            log.ODM_WARNING(str(e))

                    # Single material?
                    if args.texturing_single_material:
                        log.ODM_INFO("Packing to single material")

                        packed_dir = os.path.join(r['out_dir'], 'packed')
                        if io.dir_exists(packed_dir):
                            log.ODM_INFO("Removing old packed directory {}".format(packed_dir))
                            shutil.rmtree(packed_dir)
                        
                        try:
                            obj_pack(os.path.join(r['out_dir'], tree.odm_textured_model_obj), packed_dir, _info=log.ODM_INFO)
                            
                            # Move packed/* into texturing folder
                            system.delete_files(r['out_dir'], (".vec", ))
                            system.move_files(packed_dir, r['out_dir'])
                            if os.path.isdir(packed_dir):
                                os.rmdir(packed_dir)
                        except Exception as e:
                            log.ODM_WARNING(str(e))


                # Backward compatibility: copy odm_textured_model_geo.mtl to odm_textured_model.mtl
                # for certain older WebODM clients which expect a odm_textured_model.mtl
                # to be present for visualization
                # We should remove this at some point in the future
                geo_mtl = os.path.join(r['out_dir'], 'odm_textured_model_geo.mtl')
                if io.file_exists(geo_mtl):
                    nongeo_mtl = os.path.join(r['out_dir'], 'odm_textured_model.mtl')
                    shutil.copy(geo_mtl, nongeo_mtl)

                progress += progress_per_run
                self.update_progress(progress)
            else:
                log.ODM_WARNING('Found a valid ODM Texture file in: %s'
                                % odm_textured_model_obj)
        
        if args.optimize_disk_space:
            for r in nonloc.runs:
                if io.file_exists(r['model']):
                    os.remove(r['model'])
            
            undistorted_images_path = os.path.join(tree.opensfm, "undistorted", "images")
            if io.dir_exists(undistorted_images_path):
                shutil.rmtree(undistorted_images_path)
        

          