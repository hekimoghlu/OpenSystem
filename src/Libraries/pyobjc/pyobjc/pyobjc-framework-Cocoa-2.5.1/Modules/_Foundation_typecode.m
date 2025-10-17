/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 21, 2022.
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
/* inline definition of PyMac_GetOSType pymactoolbox.h doesn't work in 64-bit mode */

#if PY_MAJOR_VERSION == 2 && defined(USE_TOOLBOX_OBJECT_GLUE)
extern int PyMac_GetOSType(PyObject *v, OSType *pr);
extern PyObject * PyMac_BuildOSType(OSType t);

#else

static int
PyMac_GetOSType(PyObject *v, OSType *pr)
{
	uint32_t tmp;
	if (!PyBytes_Check(v) || PyBytes_Size(v) != 4) {
		PyErr_SetString(PyExc_TypeError,
			"OSType arg must be byte string of 4 chars");
		return 0;
	}
	memcpy((char *)&tmp, PyBytes_AsString(v), 4);
	*pr = (OSType)ntohl(tmp);
	return 1;
}

PyObject *
PyMac_BuildOSType(OSType t)
{
	uint32_t tmp = htonl((uint32_t)t);
	return PyBytes_FromStringAndSize((char *)&tmp, 4);
}
#endif



PyDoc_STRVAR(objc_NSFileTypeForHFSTypeCode_doc,
	"NSString *NSFileTypeForHFSTypeCode(OSType hfsTypeCode);");
static PyObject* 
objc_NSFileTypeForHFSTypeCode(
	PyObject* self __attribute__((__unused__)), 
	PyObject* args, 
	PyObject* kwds)
{
static	char* keywords[] = { "hfsTypeCode", NULL };
	PyObject*  result;
	NSString*  oc_result;
	OSType hfsTypeCode;

	if (!PyArg_ParseTupleAndKeywords(args, kwds, 
			"i:NSFileTypeForHFSTypeCode",
			keywords, &hfsTypeCode)) {
		PyErr_Clear();
		if (!PyArg_ParseTupleAndKeywords(args, kwds, 
				"O&:NSFileTypeForHFSTypeCode",
				keywords, PyMac_GetOSType, &hfsTypeCode)) {
			return NULL;
		}
	}
	
	PyObjC_DURING
		oc_result = NSFileTypeForHFSTypeCode(hfsTypeCode);
	PyObjC_HANDLER
		PyObjCErr_FromObjC(localException);
		oc_result = NULL;
	PyObjC_ENDHANDLER

	if (PyErr_Occurred()) return NULL;

	result = PyObjC_IdToPython(oc_result);
	return result;
}

PyDoc_STRVAR(objc_NSHFSTypeCodeFromFileType_doc,
		"OSType NSHFSTypeCodeFromFileType(NSString *fileType);");
static PyObject* 
objc_NSHFSTypeCodeFromFileType(
	PyObject* self __attribute__((__unused__)), 
	PyObject* args, 
	PyObject* kwds)
{
static	char* keywords[] = { "hfsTypeCode", NULL };
	NSString*  fileType;
	OSType hfsTypeCode;

	if (!PyArg_ParseTupleAndKeywords(args, kwds, 
			"O&:NSHFSTypeCodeFromFileType",
			keywords, PyObjCObject_Convert, &fileType)) {
		return NULL;
	}
	
	PyObjC_DURING
		hfsTypeCode = NSHFSTypeCodeFromFileType(fileType);
	PyObjC_HANDLER
		hfsTypeCode = 0;
		PyObjCErr_FromObjC(localException);
	PyObjC_ENDHANDLER

	if (PyErr_Occurred()) return NULL;

	return PyMac_BuildOSType(hfsTypeCode);
}

#define FOUNDATION_TYPECODE_METHODS				\
	{ 							\
		"NSFileTypeForHFSTypeCode", 			\
		(PyCFunction)objc_NSFileTypeForHFSTypeCode, 	\
		METH_VARARGS|METH_KEYWORDS, 			\
		objc_NSFileTypeForHFSTypeCode_doc		\
	},							\
	{ 							\
		"NSHFSFTypeCodeFromFileType",			\
		(PyCFunction)objc_NSHFSTypeCodeFromFileType, 	\
		METH_VARARGS|METH_KEYWORDS, 			\
		objc_NSHFSTypeCodeFromFileType_doc 		\
	},
