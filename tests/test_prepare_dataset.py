import unittest

from prepare_dataset import convert_annotation_line


class ConvertAnnotationLineTests(unittest.TestCase):
    def test_converts_visdrone_box_to_zero_based_normalized_yolo_label(self) -> None:
        label = convert_annotation_line("10,20,30,40,1,4,0,0", 100, 100)

        self.assertEqual(label, "3 0.250000 0.400000 0.300000 0.400000")

    def test_ignores_regions_and_other_categories(self) -> None:
        self.assertIsNone(convert_annotation_line("0,0,10,10,0,0,0,0", 100, 100))
        self.assertIsNone(convert_annotation_line("0,0,10,10,0,11,0,0", 100, 100))

    def test_clips_boxes_to_image_bounds(self) -> None:
        label = convert_annotation_line("-10,80,30,40,1,1,0,0", 100, 100)

        self.assertEqual(label, "0 0.100000 0.900000 0.200000 0.200000")

    def test_discards_boxes_without_visible_area(self) -> None:
        self.assertIsNone(convert_annotation_line("100,10,5,5,1,1,0,0", 100, 100))
        self.assertIsNone(convert_annotation_line("10,10,0,5,1,1,0,0", 100, 100))

    def test_rejects_malformed_annotation(self) -> None:
        with self.assertRaises(ValueError):
            convert_annotation_line("10,20,30", 100, 100)


if __name__ == "__main__":
    unittest.main()
