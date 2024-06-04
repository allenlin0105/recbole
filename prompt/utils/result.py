

def set_up_result_dir(root, split):
    result_folder = root.joinpath(split)
    result_folder.mkdir(exist_ok=True, parents=True)
    return result_folder


def get_result_filename(user_id):
    return f"{user_id}.txt"


def user_is_predicted(result_folder, user_id):
    output_file = result_folder.joinpath(get_result_filename(user_id))
    return output_file.exists()


def output_to_file(result_folder, user_id, correct_item, candidate_items, responses):
    output_file = result_folder.joinpath(get_result_filename(user_id))
    with open(output_file, "w") as fp:
        fp.write(f"Correct item: {correct_item}\n=====\n")
        for candidate_item, response in zip(candidate_items, responses):
            fp.write(f"Candidate item: {candidate_item}\n{response}\n=====\n")