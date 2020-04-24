#!/usr/bin/env python
# coding: utf-8

import os, sys, pprint, argparse

print('Importing azureml...')
from azureml.core import Workspace, Image
from azureml.core.compute import ComputeTarget
from azureml.contrib.core.compute import IotHubCompute
from azureml.contrib.core.webservice import IotWebservice, IotBaseModuleSettings
from azureml.accel import AccelContainerImage
print('Done importing azureml')

def disable_deploy_from_image():
    """Disables the actual deploy call  """
    print('Disabling IotWebservice.deploy_from_image (demo)')
    def mocked_deploy_from_image(*args, **kwargs):
        print('IotWebservice.deploy_from_image called but is disabled. Call was:')
        print(
            'IotWebservice.deploy_from_image(\n{0},\n{1})'
            .format(pprint.pformat(args), pprint.pformat(kwargs))
            )
    IotWebservice._bu_deploy_from_image = IotWebservice.deploy_from_image # Preserve the function
    IotWebservice.deploy_from_image = mocked_deploy_from_image

class Service(object):
    """Deploys a new service to the DBE at Fermilab"""

    container_config = """{
  "ExposedPorts": {
    "50051/tcp": {}
  },
  "HostConfig": {
    "Binds": [
      "/etc/hosts:/etc/hosts"
    ],
    "Privileged": true,
    "Devices": [
      {
        "PathOnHost": "/dev/catapult0",
        "PathInContainer": "/dev/catapult0"
      },
      {
        "PathOnHost": "/dev/catapult1",
        "PathInContainer": "/dev/catapult1"
      }
    ],
    "PortBindings": {
      "50051/tcp": [
        {
          "HostPort": "50051"
        }
      ]
    }
  }
}"""

    def __init__(self):
        super(Service, self).__init__()

        print('Setting up ws...')
        if not os.path.isfile('config.json'):
            raise RuntimeError(
                'No file config.json is found, which is needed to setup '
                'the workspace. Please first download the config.json from '
                'the Azure portal (https://portal.azure.com) and put it in '
                'this directory.'
                )
        self.ws = Workspace.from_config()
        self.iot_device_id = "fermi-edge"
        print(self.ws)

        self.module_name = "tquarkrn50-v200-1"
        self.image_name = "im-klijnsma-tquarkrn50-v200"
        self.port = 50051


    def create_service(self):
        iothub_compute = IotHubCompute(self.ws, self.iot_device_id)
        print('IotHubCompute:\n{0}'.format(iothub_compute))

        routes = {"route": "FROM /messages/* INTO "}

        # Here, we define the Azure ML module with the container_config options above
        aml_module = IotBaseModuleSettings(name=self.module_name, create_option=self.container_config)

        # This time, we will leave off the external module from the deployment manifest
        deploy_config = IotWebservice.deploy_configuration(
            device_id = self.iot_device_id,
            routes = routes,
            aml_module = aml_module
            )

        # Deploy from latest version of image, using module_name as your IotWebservice name
        iot_service_name = self.module_name

        # Can specify version=x, otherwise will grab latest
        image = Image(self.ws, self.image_name) 
        print('Deploying image: {0}'.format(image))

        iot_service = IotWebservice.deploy_from_image(
            self.ws,
            iot_service_name,
            image,
            deploy_config,
            iothub_compute
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--deploy', action='store_true')
    args = parser.parse_args()

    if not(args.deploy):
        disable_deploy_from_image()

    service = Service()
    service.create_service()

if __name__ == "__main__":
    main()