import unittest
import numpy
import pandas

from cml_data_tools.online_norm import to_dataframe, to_stats, update


class OnlineNormTestCase(unittest.TestCase):

    def test_to_stats(self):
        example = pandas.DataFrame(
            numpy.arange(9).reshape(3, 3),
            columns=('a', 'b', 'c')
        )
        target = pandas.DataFrame(
            [
                [3., 6., 3],
                [4., 6., 3],
                [5., 6., 3],
            ],
            index=('a', 'b', 'c'),
            columns=('mean', 'var', 'n')
        )
        stats = to_dataframe(to_stats(example))
        pandas.testing.assert_frame_equal(target, stats)

    def test_update_overlap_labels(self):
        ex0 = pandas.DataFrame(
            numpy.arange(9).reshape(3, 3),
            columns=('a', 'b', 'c')
        )
        ex1 = ex0 + 1
        example = pandas.concat((ex0, ex1))
        target = pandas.DataFrame(
            [
                [3.5, 6.25, 6],
                [4.5, 6.25, 6],
                [5.5, 6.25, 6]
            ],
            index=('a', 'b', 'c'),
            columns=('mean', 'var', 'n')
        )
        stats = to_dataframe(to_stats(example))
        pandas.testing.assert_frame_equal(target, stats)

    def test_update_disjoint_labels(self):
        ex0 = pandas.DataFrame(
            numpy.arange(9).reshape(3, 3),
            columns=('a', 'b', 'c')
        )
        ex1 = pandas.DataFrame(
            numpy.arange(9).reshape(3, 3),
            columns=('d', 'e', 'f')
        )
        target = pandas.DataFrame(
            [
                [3., 6., 3],
                [4., 6., 3],
                [5., 6., 3],
                [3., 6., 3],
                [4., 6., 3],
                [5., 6., 3],
            ],
            index=('a', 'b', 'c', 'd', 'e', 'f'),
            columns=('mean', 'var', 'n'),
        )
        stats = to_dataframe(update(to_stats(ex0), to_stats(ex1)))
        pandas.testing.assert_frame_equal(target, stats)

    def test_update_partial_overlap(self):
        ex0 = pandas.DataFrame(
            numpy.arange(9).reshape(3, 3),
            columns=('a', 'b', 'c')
        )
        ex1 = pandas.DataFrame(
            numpy.arange(9).reshape(3, 3),
            columns=('b', 'c', 'd')
        )
        target = pandas.DataFrame(
            [
                [3.0, 6.00, 3],
                [3.5, 6.25, 6],
                [4.5, 6.25, 6],
                [5.0, 6.00, 3],
            ],
            index=('a', 'b', 'c', 'd'),
            columns=('mean', 'var', 'n'),
        )
        stats = to_dataframe(update(to_stats(ex0), to_stats(ex1)))
        pandas.testing.assert_frame_equal(target, stats)


if __name__ == '__main__':
    unittest.main()
