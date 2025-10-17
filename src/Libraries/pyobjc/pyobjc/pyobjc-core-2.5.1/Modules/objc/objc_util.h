/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 3, 2021.
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

#ifndef OBJC_UTIL
#define OBJC_UTIL

#include <Foundation/NSException.h>
#define THREADSTATE_AUTORELEASEPOOL "__threadstate_autoreleasepool"

extern PyObject* PyObjCExc_Error;
extern PyObject* PyObjCExc_NoSuchClassError;
extern PyObject* PyObjCExc_InternalError;
extern PyObject* PyObjCExc_UnInitDeallocWarning;
extern PyObject* PyObjCExc_ObjCRevivalWarning;
extern PyObject* PyObjCExc_LockError;
extern PyObject* PyObjCExc_BadPrototypeError;
extern PyObject* PyObjCExc_UnknownPointerError;

int PyObjCUtil_Init(PyObject* module);

void PyObjCErr_FromObjC(NSException* localException);
void PyObjCErr_ToObjC(void) __attribute__((__noreturn__));

void PyObjCErr_ToObjCWithGILState(PyGILState_STATE* state) __attribute__((__noreturn__));

NSException* PyObjCErr_AsExc(void);

PyObject* PyObjC_CallPython(id self, SEL selector, PyObject* arglist, BOOL* isAlloc, BOOL* isCFAlloc);

char* PyObjCUtil_Strdup(const char* value);

#include <Foundation/NSMapTable.h>
extern NSMapTableKeyCallBacks PyObjCUtil_PointerKeyCallBacks;
extern NSMapTableValueCallBacks PyObjCUtil_PointerValueCallBacks;

extern NSMapTableKeyCallBacks PyObjCUtil_ObjCIdentityKeyCallBacks;
extern NSMapTableValueCallBacks PyObjCUtil_ObjCValueCallBacks;

void    PyObjC_FreeCArray(int, void*);
int     PyObjC_PythonToCArray(BOOL, BOOL, const char*, PyObject*, void**, Py_ssize_t*, PyObject**);
PyObject* PyObjC_CArrayToPython(const char*, void*, Py_ssize_t);
PyObject* PyObjC_CArrayToPython2(const char*, void*, Py_ssize_t, bool already_retained, bool already_cfretained);
int     PyObjC_IsPythonKeyword(const char* word);


extern int PyObjCRT_SimplifySignature(const char* signature, char* buf, size_t buflen);

extern int PyObjCObject_Convert(PyObject* object, void* pvar);
extern int PyObjCClass_Convert(PyObject* object, void* pvar);

extern int PyObjC_is_ascii_string(PyObject* unicode_string, const char* ascii_string);
extern int PyObjC_is_ascii_prefix(PyObject* unicode_string, const char* ascii_string, size_t n);

extern PyObject* PyObjC_ImportName(const char* name);

#endif /* OBJC_UTIL */
