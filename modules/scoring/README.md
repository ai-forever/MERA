# MERA scoring
## Generate sample submission
```shell
python generage_sample_submission.py
```

Arguments:
* `submission_path` - path to submission (in `zip` format)
* `produce_errors` - if need add errors to submission

## Call evaluation script
Since there are no correct answers to all tasks in the public data, we only show our assessment code here.
The resulting quality metrics are not representative.

```shell
python evaluate_submission.py
```

Arguments:
* `submission_path` - path to submission (in `zip` format)
* `results_path` - path to store results

For run need `gpu` or modify config `configs/rudetox.yaml`: field `device` change to `cpu`.

## Description of response
Response:

```json
{
    "id": "<id>",
    "status": "Failed",
    "error_reason": {"<task_name>": [{"type": "<error_type>", "comment": "<comment>"}]},
    "global_error_reason": [{"type": "<error_type>", "comment": "<comment>"}],
    "results": {
           "<task_name>": {"<metric_name>": "<metric_value>"}
    }
}
```

Field `results` may be empty or partially contain evaluation results (for files that had no errors).

Field `error_reason` (if present) contains errors, for each task separately.

The `global_error_reason` field (if present) contains global errors.

Errors may be the following:

* `unreadable_zip` - can't read zip.

Error list item format:

```
{"type": "unreadable_zip", "comment": "<comment>"}
```

* `no_task` - there is no task file in the submission (if you sent 1 file, there will be no such error).
also then the validation for the presence of the task will be on your side. it's just looking at the intersection of file names with the list we'll give.

Error list item format:

```
{"type": "no_task", "comment": "<comment>"}
```
* `extension` - incorrect file extension.

Error list item format:

```
{"type": "extension", "extension": extension, "comment": "<comment>"}
```

* `unreadable_file` - the file cannot be read.

Error list item format:

```
{"type": "unreadable_file", , "comment": "<comment>"}
```

* `no_data_field` - there is no `"data"` key in the loaded `json`.

Error list item format:

```
{"type": "no_data_field", "comment": "<comment>"}
```

* `no_split` - there is no `split` key in the `"data"` field. The default value of the `split` key is `test`.

Error list item format:

```
{"type": "no_split", "split": "<split>", "comment": "<comment>"}
```

* `no_outputs_field_for_doc` - there is no `outputs` key in a specific document.

Error list item format:

```
{"type": "no_outputs_field_for_doc", "example_number": idx, "comment": "<comment>"}
```

Here `example_number` is the number of the example in the split in the document.

* `no_meta_field_for_doc` - there is no `meta` key in a specific document.

Error list item format:

```
{"type": "no_meta_field_for_doc", "example_number": idx, "comment": "<comment>"}
```

Here `example_number` is the number of the example in the split in the document.

* `no_id_field_for_doc` - there is no `id` key in the `meta` field in a specific document.

Error list item format:

```
{"type": "no_id_field_for_doc", "example_number": idx, "comment": "<comment>"}
```

Here `example_number` is the number of the example in the split in the document.

* `no_id` - there is no example with `id` in the submit.

Error list item format:

```
{"type": "no_id", "doc_id": doc_id, "comment": "<comment>"}
```

* `doc_output_type_error` - The response type does not match the expected response type.

Error list item format:

```
{"type": doc_output_type_error, "doc_id": doc_id, "comment": "<comment>"}
```

* `doc_parse_output_error` - error in receiving a response for a specific example from a submission.

Error list item format:

```
{"type": "doc_parse_output_error", "doc_id": doc_id, "comment": "<comment>"}
```
