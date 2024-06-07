from enum import Enum


class EnumBase(Enum):
    def __str__(self):
        return f"{self.value}"

    def __eq__(self, other):
        return str(self) == str(other)


class Errors(EnumBase):
    no_task = "no_task"
    unreadable_file = "unreadable_file"
    unreadable_zip = "unreadable_zip"
    extension = "extension"
    s3_error = "s3_error"
    no_data_field = "no_data_field"
    no_split = "no_split"
    no_outputs_field_for_doc = "no_outputs_field_for_doc"
    no_meta_field_for_doc = "no_meta_field_for_doc"
    no_id_field_for_doc = "no_id_field_for_doc"
    no_id = "no_id"
    doc_output_type_error = "doc_output_type_error"
    doc_parse_output_error = "doc_parse_output_error"
    task_system_error = "task_system_error"


class SubmissionStatus(EnumBase):
    ok = "OK"
    failed = "Failed"
