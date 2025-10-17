/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 22, 2025.
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
#ifndef _objc_support_H
#define _objc_support_H

extern BOOL PyObjC_signatures_compatible(const char* type1, const char* type2);
/* Returns True iff two typestrings are compatible:
 * - elements have same size
 * - 'id' is compatible with 'void*' and not with other types
 * - 'float'/'double' are not compatible with integer types.
 */

/*#F Takes a C value pointed by @var{datum} with its type encoded in
  @var{type}, that should be coming from an ObjC @encode directive,
  and returns an equivalent Python object where C structures and
  arrays are represented as tuples. */
extern PyObject *pythonify_c_value (const char *type,
				    void *datum);
extern PyObject *pythonify_c_return_value (const char *type,
				    void *datum);

extern PyObject *pythonify_c_array_nullterminated(const char* type, void* datum, BOOL already_retained, BOOL already_cfretained);

extern int depythonify_c_array_count(const char* type, Py_ssize_t count, BOOL strict, PyObject* value, void* datum, BOOL already_retained, BOOL already_cfretained);
extern Py_ssize_t c_array_nullterminated_size(PyObject* object, PyObject** seq);
extern int depythonify_c_array_nullterminated(const char* type, Py_ssize_t count, PyObject* value, void* datum, BOOL already_retained, BOOL already_cfretained);

/*#F Takes a Python object @var{arg} and translate it into a C value
  pointed by @var{datum} accordingly with the type specification
  encoded in @var{type}, that should be coming from an ObjC @encode
  directive.
  Returns NULL on success, or a static error string describing the
  error. */
extern int depythonify_c_value (const char *type,
					PyObject *arg,
					void *datum);
extern int depythonify_c_return_value (const char *type,
					PyObject *arg,
					void *datum);

extern int depythonify_c_return_array_count(const char* rettype, Py_ssize_t count, PyObject* arg, void* resp, BOOL already_retained, BOOL already_cfretained);
extern int depythonify_c_return_array_nullterminated(const char* rettype, PyObject* arg, void* resp, BOOL already_retained, BOOL already_cfretained);


extern Py_ssize_t PyObjCRT_SizeOfReturnType(const char* type);
extern Py_ssize_t PyObjCRT_SizeOfType(const char *type);
extern Py_ssize_t PyObjCRT_AlignOfType(const char *type);
extern const char *PyObjCRT_SkipTypeSpec (const char *type);
extern const char* PyObjCRT_NextField(const char *type);
extern const char* PyObjCRT_SkipTypeQualifiers (const char* type);
extern Py_ssize_t PyObjCRT_AlignedSize (const char *type);


extern const char* PyObjCRT_RemoveFieldNames(char* buf, const char* type);

/*
 * Compatibility with pyobjc-api.h
 */
static inline id PyObjC_PythonToId(PyObject* value)
{
	id res;
	int r;

	r = depythonify_c_value(@encode(id), value, &res);
	if (r == -1) {
		return NULL;
	} else {
		return res;
	}
}

static inline PyObject* PyObjC_IdToPython(id value)
{
	PyObject* res;

	res = pythonify_c_value(@encode(id), &value);
	return res;
}

#endif /* _objc_support_H */
