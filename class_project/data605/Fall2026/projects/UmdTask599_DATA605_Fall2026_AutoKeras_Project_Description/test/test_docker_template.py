"""
Import as:

import class_project.project_template.test.test_docker_all as tptdal
"""

import logging

import pytest

import helpers.hdocker_tests as hdoctest

_LOG = logging.getLogger(__name__)


# #############################################################################
# Test_docker
# #############################################################################


class Test_docker(hdoctest.DockerTestCase):
    """
    Run all Docker tests for `class_project/UmdTask599_DATA605_Fall2026_AutoKeras_Project_Description`.
    """

    _test_file = __file__

    @pytest.mark.slow
    def test1(self) -> None:
        """
        Test that `autokeras.example.ipynb` runs without error inside Docker.
        """
        # Prepare inputs.
        notebook_name = "autokeras.example.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=True, generate_html=False
        )

    @pytest.mark.slow
    def test2(self) -> None:
        """
        Test that `autokeras.API.ipynb` runs without error inside Docker.
        """
        # Prepare inputs.
        notebook_name = "autokeras.API.ipynb"
        # Run test.
        self.run_notebook(
            notebook_name, use_docker_cmd=True, generate_html=False
        )
