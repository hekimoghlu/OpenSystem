/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 16, 2023.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
/*
 * Provides a stream-based API for generating JSON output
 *
 * Handles tedious tasks like worrying about comma placement (and avoiding trailing commas).
 * Assumes strings are already escaped (if necessary) and does no error checking (thus it
 * may produce invalid JSON when used improperly).
 *
 * As a convenience, when the provided `json` stream is NULL (i.e. it was never initialized
 * by `JSON_OPEN`) these APIs will do nothing.
 *
 * Example usage:
 *
 *  JSON_t json = JSON_OPEN("/path/to/output.json")
 *  JSON_OBJECT_BEGIN(json); // root object
 *
 *  JSON_OBJECT_SET(json, version, %.1f, 1.0);
 *  JSON_OBJECT_SET_BOOL(json, has_fruit, 1);
 *
 *  // Note the required quotes for strings (formatted or not)
 *  char *mystr = "hello";
 *  JSON_OBJECT_SET(json, formatted_string, "%s", mystr);
 *  JSON_OBJECT_SET(json, literal_string, "my literal string");
 *
 *  JSON_KEY(json, fruit_array);
 *  JSON_ARRAY_BEGIN(json); // fruit_array
 *  JSON_ARRAY_APPEND(json, "my literal string");
 *  JSON_ARRAY_APPEND(json, "<0x%08llx>", 0xface);
 *  JSON_ARRAY_APPEND(json, %d, 3);
 *  JSON_ARRAY_END(json); // fruit_array
 *
 *  JSON_OBJECT_END(json); // root object
 *  JSON_CLOSE(json);
 */

#ifndef _JSON_H_
#define _JSON_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define _JSON_IF(json, code) \
  if (json != NULL) { \
    code; \
  }
#define _JSON_COMMA(json) \
  if (json->require_comma) { \
    fprintf(json->stream, ","); \
  }

struct _JSON {
  FILE* stream;
  bool require_comma;
};
typedef struct _JSON * JSON_t;

#pragma mark Open/Close
/* Return a new JSON_t stream */
static inline JSON_t JSON_OPEN(const char *path) {
  JSON_t p = malloc(sizeof(struct _JSON));
  p->stream = fopen(path, "w+");
  p->require_comma = false;
  return p;
}

/* Close an existing JSON stream, removing trailing commas */
#define JSON_CLOSE(json) _JSON_IF(json, fclose(json->stream); free(json))

#pragma mark Keys/Values
/* Output the `key` half of a key/value pair */
#define JSON_KEY(json, key) _JSON_IF(json, _JSON_COMMA(json); fprintf(json->stream, "\"" #key "\":"); json->require_comma = false)
/* Output the `value` half of a key/value pair */
#define JSON_VALUE(json, format, ...) _JSON_IF(json, fprintf(json->stream, #format, ##__VA_ARGS__); json->require_comma = true)

#define _JSON_BEGIN(json, character) _JSON_COMMA(json); fprintf(json->stream, #character); json->require_comma = false;
#define _JSON_END(json, character) fprintf(json->stream, #character); json->require_comma = true;
#define _JSON_BOOL(val) ( val ? "true" : "false" )

#pragma mark Objects
/* Start a new JSON object */
#define JSON_OBJECT_BEGIN(json) _JSON_IF(json, _JSON_BEGIN(json, {))
/* Set a value in the current JSON object */
#define JSON_OBJECT_SET(json, key, format, ...) _JSON_IF(json, JSON_KEY(json, key); JSON_VALUE(json, format, ##__VA_ARGS__))
/* Set a boolean in the current JSON object */
#define JSON_OBJECT_SET_BOOL(json, key, value) JSON_OBJECT_SET(json, key, %s, _JSON_BOOL(value))
/* End the current JSON object */
#define JSON_OBJECT_END(json) _JSON_IF(json, _JSON_END(json, }))

#pragma mark Arrays
/* Start a new JSON array */
#define JSON_ARRAY_BEGIN(json) _JSON_IF(json, _JSON_BEGIN(json, [))
/* Append a value to the current JSON array */
#define JSON_ARRAY_APPEND(json, format, ...) _JSON_IF(json, _JSON_COMMA(json); JSON_VALUE(json, format, ##__VA_ARGS__))
/* End the current JSON array */
#define JSON_ARRAY_END(json) _JSON_IF(json, _JSON_END(json, ]))

#endif /* _JSON_H_ */
