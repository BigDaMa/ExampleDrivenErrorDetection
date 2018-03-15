import numpy as np

class ReplaceError:

	@staticmethod
	def get_error_ids(column_data, error_fraction):
		indices = range(len(column_data))
		np.random.shuffle(indices)
		error_indices = indices[0:int(len(column_data) * error_fraction)]

		return error_indices

	@staticmethod
	def error(column_data, error_fraction):

		print column_data

		error_indices = ReplaceError.get_error_ids(column_data, error_fraction)

		chars = "abcdefgh1234567890"

		for error_id in error_indices:
			char_to_replace = np.random.randint(len(column_data[error_id]))
			char_error = None
			while char_error == None:
				try_char = np.random.randint(len(chars))
				if chars[try_char] != column_data[error_id][char_to_replace]:
					char_error = chars[try_char]

			new_str = ""
			for i_str in range(len(column_data[error_id])):
				if i_str == char_to_replace:
					new_str += char_error
				else:
					new_str += column_data[error_id][i_str]

			column_data[error_id] = new_str



		return column_data

