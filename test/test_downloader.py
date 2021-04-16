import os
import unittest

from downloader import download_data


class DataDownloadingTest(unittest.TestCase):
    """Tests for function :func download_data:
    """

    def test_parameter_checking(self):
        """Test if wrong parameters are recognised as wrong
        """
        params = ['12-2015', '12-15', '2015_12', '2015.12', '12.2015', '12/2015', '12/15', '2015-00', '2015-13']
        # list of examples of possible wrong month parameters
        for month in params:
            with self.assertRaises(ValueError) as cm:  # ValueError should be raised with proper message
                download_data(month)
            self.assertEqual('Month parameter should be in form `yyyy-mm`', cm.exception.args[0], msg=month)
            # check if error message was as expected

    def test_existence_checking(self):
        """Tests wheather not existing datasets are marked as not existing
        and wheather proper month parameter wouldn't be consider inproper
        """
        params = [f'2010-{num:0>2}' for num in range(1, 13)]  # all month of 2010
        for month in params:
            with self.assertRaises(ValueError) as cm:  # ValueError should be raised with proper message
                download_data(month)
            self.assertEqual(f'Dataset from {month} cannot be found on lichess.org', cm.exception.args[0], msg=month)
            # check if error message was as expected

    def test_downloading(self):
        """Tests wheather file was saved to returned location
        """
        month = '2013-01'  # smallest of available datasets
        path = download_data(month)
        self.assertTrue(os.path.isfile(path), msg='File on returned location does not exist')
        os.remove(path)


if __name__ == '__main__':
    unittest.main()
