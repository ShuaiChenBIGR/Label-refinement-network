from collections import OrderedDict
import numpy as np
import random
import csv
import os

_EPS = 1.0e-10


def get_vector_two_points(begin_point, end_point):
    return (end_point[0] - begin_point[0],
            end_point[1] - begin_point[1],
            end_point[2] - begin_point[2])


def get_norm_vector(in_vector):
    return np.linalg.norm(in_vector)


def get_distance_two_points(begin_point, end_point):
    return get_norm_vector(get_vector_two_points(begin_point, end_point))


def get_point_inside_segment(begin_point, end_point, rel_dist_segm):
    vector_segment = get_vector_two_points(begin_point, end_point)
    return (begin_point[0] + rel_dist_segm * vector_segment[0],
            begin_point[1] + rel_dist_segm * vector_segment[1],
            begin_point[2] + rel_dist_segm * vector_segment[2])


def _get_indexes_canditate_inside_blank(point_center, max_dist_2cen, image_size):
    min_index_x = int(np.floor(point_center[0] - max_dist_2cen))
    max_index_x = int(np.ceil(point_center[0] + max_dist_2cen))
    min_index_y = int(np.floor(point_center[1] - max_dist_2cen))
    max_index_y = int(np.ceil(point_center[1] + max_dist_2cen))
    min_index_z = int(np.floor(point_center[2] - max_dist_2cen))
    max_index_z = int(np.ceil(point_center[2] + max_dist_2cen))
    min_index_x = max(min_index_x, 0)
    max_index_x = min(max_index_x, image_size[0] - 1)
    min_index_y = max(min_index_y, 0)
    max_index_y = min(max_index_y, image_size[1] - 1)
    min_index_z = max(min_index_z, 0)
    max_index_z = min(max_index_z, image_size[2] - 1)
    indexes_x = np.arange(min_index_x, max_index_x + 1)
    indexes_y = np.arange(min_index_y, max_index_y + 1)
    indexes_z = np.arange(min_index_z, max_index_z + 1)
    return np.stack(np.meshgrid(indexes_x, indexes_y, indexes_z, indexing='ij'), axis=3)


def generate_error_blank_branch(inout_mask, point_center, vector_axis, diam_base, length_axis):
    norm_vector_axis = np.sqrt(np.dot(vector_axis, vector_axis))
    unit_vector_axis = np.array(vector_axis) / norm_vector_axis
    radius_base = diam_base / 2.0
    half_length_axis = length_axis / 2.0
    image_size = inout_mask.shape[::-1] # get correct format (dx, dy, dz)

    # candidates: subset of all possible indexes where to check condition for blank shape -> to save time
    dist_corner_2center = np.sqrt(radius_base ** 2 + half_length_axis ** 2)
    indexes_candits_inside_blank = _get_indexes_canditate_inside_blank(point_center, dist_corner_2center, image_size)
    # array of indexes, with dims [num_indexes_x, num_indexes_y, num_indexes_z, 3]

    # relative position of candidate indexes to center
    points_rel2center_candits_inside = indexes_candits_inside_blank - point_center

    # distance to center, parallel to axis -> dot product of distance vectors with 'vector_axis'
    dist_rel2center_parall_axis_candits = np.dot(points_rel2center_candits_inside, unit_vector_axis)

    # distance to center, perpendicular to axis -> Pythagoras (distance_2center ^2 - distance_2center_parall_axis ^2)
    dist_rel2center_perpen_axis_candits = np.sqrt(np.square(np.linalg.norm(points_rel2center_candits_inside, axis=3))
                                                  - np.square(dist_rel2center_parall_axis_candits) + _EPS)

    # conditions for cylinder: 1) distance to center, parallel to axis, is less than 'half_length_axis'
    #                          2) distance to center, perpendicular to axis, is less than 'radius_base'
    is_indexes_inside_blank_cond1 = np.abs(dist_rel2center_parall_axis_candits) <= half_length_axis
    is_indexes_inside_blank_cond2 = np.abs(dist_rel2center_perpen_axis_candits) <= radius_base

    is_indexes_inside_blank = np.logical_and(is_indexes_inside_blank_cond1, is_indexes_inside_blank_cond2)
    # array of ['True', 'False'], with 'True' for indexes that are inside the blank

    indexes_inside_blank = indexes_candits_inside_blank[is_indexes_inside_blank]

    # blank error: set '0' to voxels for indexes inside the blank
    (indexes_x_in, indexes_y_in, indexes_z_in) = np.transpose(indexes_inside_blank)
    inout_mask[indexes_z_in, indexes_y_in, indexes_x_in] = 0

    return inout_mask


class CsvFileReader(object):

    @staticmethod
    def get_data_type(in_value_str):
        if in_value_str.isdigit():
            if in_value_str.count(' ') > 1:
                return 'group_integer'
            else:
                return 'integer'
        elif in_value_str.replace('.', '', 1).isdigit() and in_value_str.count('.') < 2:
            return 'float'
        else:
            return 'string'

    @classmethod
    def get_data(cls, input_file):
        with open(input_file, 'r') as fin:
            csv_reader = csv.reader(fin, delimiter=',')

            list_fields = next(csv_reader)  # read header
            list_fields = [elem.lstrip() for elem in list_fields]  # remove empty leading spaces ' '

            # output data as dictionary (key: field name, value: field data column)
            out_dict_data = OrderedDict([(ifield, []) for ifield in list_fields])

            num_fields = len(list_fields)
            for irow, row_data in enumerate(csv_reader):
                row_data = [elem.lstrip() for elem in row_data]  # remove empty leading spaces ' '

                if irow == 0:
                    # get the data type for each field
                    list_datatype_fields = []
                    for ifie in range(num_fields):
                        in_value_str = row_data[ifie]
                        in_data_type = cls.get_data_type(in_value_str)
                        list_datatype_fields.append(in_data_type)

                for ifie in range(num_fields):
                    field_name = list_fields[ifie]
                    in_value_str = row_data[ifie]
                    in_data_type = list_datatype_fields[ifie]

                    if in_value_str == 'NaN' or in_value_str == 'nan':
                        out_value = np.NaN
                    elif in_data_type == 'integer':
                        out_value = int(in_value_str)
                    elif in_data_type == 'group_integer':
                        out_value = tuple([int(elem) for elem in in_value_str.split(' ')])
                    elif in_data_type == 'float':
                        out_value = float(in_value_str)
                    else:
                        out_value = in_value_str

                    out_dict_data[field_name].append(out_value)

        return out_dict_data


class RandomAirwayErrorGenerator(object):
    """Generate random errors in random branches in the airway segmentation mask

    Args:
        is_error_type1 (bool): option to enable errors type1: blanking small regions in random branches
        p_branches_error_type1 (float): proportion of branches where to generate errors type 1
        is_error_type2 (bool): option to enable errors type1: blanking small regions in random branches
        p_branches_error_type2 (float): proportion of branches where to generate errors type 2
    """

    _is_exclude_small_branches_error_t1 = True
    _min_length_branch_candits_error_t1 = 6.0
    _min_generation_error_t1 = 3
    _inflate_diam_error_t1 = 4.0
    _max_diam_error_t1 = 30.0
    _min_length_error_t1 = 2.0
    _inflate_diam_error_t2 = 6.0
    _max_diam_error_t2 = 30.0

    def __init__(self, dict_measures_files,
                 is_error_type1, p_branches_error_type1,
                 is_error_type2, p_branches_error_type2):
        self._is_error_type1 = is_error_type1
        self._p_branches_error_type1 = p_branches_error_type1
        self._error_type1_threshold = p_branches_error_type1
        self._is_error_type2 = is_error_type2
        self._p_branches_error_type2 = p_branches_error_type2
        self._error_type2_threshold = p_branches_error_type2

        self._dict_measures_data = OrderedDict()
        self._dict_indexes_candidate_branches_errT1 = OrderedDict()
        self._dict_indexes_candidate_branches_errT2 = OrderedDict()

        for i_patient, i_measures_file in dict_measures_files.items():
            i_measures_data = self._load_data_measures(i_measures_file)
            self._dict_measures_data[i_patient] = i_measures_data

            i_indexes_candidate_branches_errT1 = self._calc_indexes_candidate_branches_error_type1(i_patient)
            i_indexes_candidate_branches_errT2 = self._calc_indexes_candidate_branches_error_type2(i_patient)
            self._dict_indexes_candidate_branches_errT1[i_patient] = i_indexes_candidate_branches_errT1
            self._dict_indexes_candidate_branches_errT2[i_patient] = i_indexes_candidate_branches_errT2

    def _load_data_measures(self, input_file):
        list_fields_keep_measure_file = ['d_inner_global', 'airway_length', 'generation', 'childrenID', 'begPoint_x',
                                         'endPoint_x', 'begPoint_y', 'endPoint_y', 'begPoint_z', 'endPoint_z']
        in_measures_data = CsvFileReader.get_data(input_file)
        out_data = OrderedDict()
        for ifield in list_fields_keep_measure_file:
            out_data[ifield] = np.array(in_measures_data[ifield])
        return out_data

    @staticmethod
    def _initialise_random_seed():
        seed = random.randint(1, 1000000)
        np.random.seed(seed + 1)

    def _calc_indexes_candidate_branches_error_type1(self, ipatient):
        # Get indexes of candidate branches where to generate errors of type 1:
        # branches of generation higher than 3, and (if chosen) branches larger than a threshold

        generation_branches = self._dict_measures_data[ipatient]['generation']
        begpoint_x_branches = self._dict_measures_data[ipatient]['begPoint_x']
        endpoint_x_branches = self._dict_measures_data[ipatient]['endPoint_x']
        begpoint_y_branches = self._dict_measures_data[ipatient]['begPoint_y']
        endpoint_y_branches = self._dict_measures_data[ipatient]['endPoint_y']
        begpoint_z_branches = self._dict_measures_data[ipatient]['begPoint_z']
        endpoint_z_branches = self._dict_measures_data[ipatient]['endPoint_z']
        num_branches = len(begpoint_x_branches)

        # 1st: exclude branches of length shorter than '_min_length_branch_candits_error_t1'
        if self._is_exclude_small_branches_error_t1:
            indexes_excluded_branches = []
            for ibrh in range(num_branches):
                begin_point_branch = (begpoint_x_branches[ibrh], begpoint_y_branches[ibrh], begpoint_z_branches[ibrh])
                end_point_branch = (endpoint_x_branches[ibrh], endpoint_y_branches[ibrh], endpoint_z_branches[ibrh])
                length_branch = get_distance_two_points(begin_point_branch, end_point_branch)

                if length_branch < self._min_length_branch_candits_error_t1:
                    indexes_excluded_branches.append(ibrh)
        else:
            indexes_excluded_branches = []

        # 2nd: exclude the larger main branches (with lowest generation number)
        indexes_excluded_branches_more = [ibrh for ibrh, igen in enumerate(generation_branches) if igen < self._min_generation_error_t1]
        indexes_excluded_branches += indexes_excluded_branches_more

        indexes_all_branches = list(range(num_branches))
        indexes_included_branches = [ibrh for ibrh in indexes_all_branches if ibrh not in indexes_excluded_branches]

        return indexes_included_branches

    def _calc_indexes_candidate_branches_error_type2(self, ipatient):
        # Get indexes of candidate branches where to generate errors of type 2: the terminal branches

        in_children_id_branches = self._dict_measures_data[ipatient]['childrenID']

        indexes_terminal_branches = [ibrh for ibrh, child_ids_brhs in enumerate(in_children_id_branches) if child_ids_brhs == '']

        return indexes_terminal_branches

    def __call__(self, in_labels, ipatient):

        self._initialise_random_seed()

        out_label_errors = in_labels.copy()

        eps = 0.01
        self._p_branches_error_type1 = random.uniform(eps, self._error_type1_threshold - eps)
        self._p_branches_error_type2 = random.uniform(eps, self._error_type2_threshold - eps)

        if self._is_error_type1:
            out_label_errors = self._generate_errors_type1(out_label_errors, ipatient)

        if self._is_error_type2:
            out_label_errors = self._generate_errors_type2(out_label_errors, ipatient)

        return out_label_errors

    def _generate_errors_type1(self, inout_label_errors, ipatient):
        "Type 1: Blanking small regions in random branches"

        diameter_branches = self._dict_measures_data[ipatient]['d_inner_global']
        generation_branches = self._dict_measures_data[ipatient]['generation']
        begpoint_x_branches = self._dict_measures_data[ipatient]['begPoint_x']
        endpoint_x_branches = self._dict_measures_data[ipatient]['endPoint_x']
        begpoint_y_branches = self._dict_measures_data[ipatient]['begPoint_y']
        endpoint_y_branches = self._dict_measures_data[ipatient]['endPoint_y']
        begpoint_z_branches = self._dict_measures_data[ipatient]['begPoint_z']
        endpoint_z_branches = self._dict_measures_data[ipatient]['endPoint_z']
        indexes_candidate_branches = self._dict_indexes_candidate_branches_errT1[ipatient]

        num_candits_branches = len(indexes_candidate_branches)
        num_branches_error = int(self._p_branches_error_type1 * num_candits_branches)

        if num_branches_error == 0:
            return inout_label_errors

        # As sample probability, use AIRWAY GENERATION NUMBER, so that terminal branches have more likely errors
        sample_probs_generation = [generation_branches[ibrh] - self._min_generation_error_t1 + 1 for ibrh in indexes_candidate_branches]
        sample_probs_generation = np.array(sample_probs_generation) / np.sum(sample_probs_generation)

        indexes_branches_generate_error = np.random.choice(indexes_candidate_branches, num_branches_error, replace=False, p=sample_probs_generation)
        indexes_branches_generate_error = np.sort(indexes_branches_generate_error)

        for ibrh in indexes_branches_generate_error:
            begin_point_branch = (begpoint_x_branches[ibrh], begpoint_y_branches[ibrh], begpoint_z_branches[ibrh])
            end_point_branch = (endpoint_x_branches[ibrh], endpoint_y_branches[ibrh], endpoint_z_branches[ibrh])
            diameter_branch = diameter_branches[ibrh]

            vector_axis_branch = get_vector_two_points(begin_point_branch, end_point_branch)
            length_branch = get_norm_vector(vector_axis_branch)

            # position center blank: random along the branch
            rel_pos_center_blank = np.random.random()
            loc_center_blank = get_point_inside_segment(begin_point_branch, end_point_branch, rel_pos_center_blank)

            # diameter base blank: the branch diameter (inflated several times)
            diam_base_blank = self._inflate_diam_error_t1 * diameter_branch
            diam_base_blank = min(diam_base_blank, self._max_diam_error_t1)

            # length blank: random between min. (1 voxel) and the branch length
            length_axis_blank = np.random.random() * length_branch
            length_axis_blank = max(length_axis_blank, self._min_length_error_t1)

            inout_label_errors = generate_error_blank_branch(inout_label_errors, loc_center_blank, vector_axis_branch, diam_base_blank, length_axis_blank)

        return inout_label_errors

    def _generate_errors_type2(self, inout_label_errors, ipatient):
        "Type 2: Blanking partially random (most of) terminal branches"

        diameter_branches = self._dict_measures_data[ipatient]['d_inner_global']
        begpoint_x_branches = self._dict_measures_data[ipatient]['begPoint_x']
        endpoint_x_branches = self._dict_measures_data[ipatient]['endPoint_x']
        begpoint_y_branches = self._dict_measures_data[ipatient]['begPoint_y']
        endpoint_y_branches = self._dict_measures_data[ipatient]['endPoint_y']
        begpoint_z_branches = self._dict_measures_data[ipatient]['begPoint_z']
        endpoint_z_branches = self._dict_measures_data[ipatient]['endPoint_z']
        indexes_candidate_branches = self._dict_indexes_candidate_branches_errT2[ipatient]

        num_candits_branches = len(indexes_candidate_branches)
        num_branches_error = int(self._p_branches_error_type2 * num_candits_branches)

        if num_branches_error == 0:
            return inout_label_errors

        indexes_branches_generate_error = np.random.choice(indexes_candidate_branches, num_branches_error, replace=False)
        indexes_branches_generate_error = np.sort(indexes_branches_generate_error)

        for ibrh in indexes_branches_generate_error:
            begin_point_branch = (begpoint_x_branches[ibrh], begpoint_y_branches[ibrh], begpoint_z_branches[ibrh])
            end_point_branch = (endpoint_x_branches[ibrh], endpoint_y_branches[ibrh], endpoint_z_branches[ibrh])
            diameter_branch = diameter_branches[ibrh]

            vector_axis_branch = get_vector_two_points(begin_point_branch, end_point_branch)
            length_branch = get_norm_vector(vector_axis_branch)

            # position center blank: random in the first half of the branch
            rel_pos_begin_blank = np.random.random() * 0.5
            rel_pos_center_blank = (rel_pos_begin_blank + 1.0) / 2.0
            loc_center_blank = get_point_inside_segment(begin_point_branch, end_point_branch, rel_pos_center_blank)

            # diameter base blank: the branch diameter (inflated several times)
            diam_base_blank = self._inflate_diam_error_t2 * diameter_branch
            diam_base_blank = min(diam_base_blank, self._max_diam_error_t2)

            # length blank: distance between start blank and end of the branch
            length_axis_blank = (1.0 - rel_pos_begin_blank) * length_branch

            inout_label_errors = generate_error_blank_branch(inout_label_errors, loc_center_blank, vector_axis_branch, diam_base_blank, length_axis_blank)

        return inout_label_errors
