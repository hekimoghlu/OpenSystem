/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 26, 2021.
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
#include "PWOSequence.h"
#include "PWOMSequence.h"
#include "PWOMapping.h"
#include "PWOCallable.h"

// dummy exception singleton
const PWDException PWDPyExceptionObj;
const PWDException &PWDPyException = PWDPyExceptionObj;

// incref new owner, and decref old owner, and adjust to new owner
void PWOBase::GrabRef(PyObject *newObj) {
  // be careful to incref before decref if old is same as new
  Py_XINCREF(newObj);
  Py_XDECREF(_own);
  _own = _obj = newObj;
}

PWOTuple::PWOTuple(const PWOList &list): PWOSequence(PyList_AsTuple(list)) {
  LoseRef(_obj);
}

PWOListMmbr::PWOListMmbr(PyObject *obj, PWOList &parent, int ndx): PWOBase(obj),
  _parent(parent), _ndx(ndx){}

PWOListMmbr &PWOListMmbr::operator = (const PWOBase &other) {
  GrabRef(other);
  //Py_XINCREF(_obj); // this one is for setItem to steal
  _parent.setItem(_ndx,  *this);
  return  *this;
}

PWOMappingMmbr &PWOMappingMmbr::operator = (const PWOBase &other) {
  GrabRef(other);
  _parent.setItem(_key,  *this);
  return  *this;
}

PWOBase PWOCallable::call()const {
  static PWOTuple _empty;
  PyObject *rslt = PyEval_CallObjectWithKeywords(*this, _empty, NULL);
  if (rslt == 0)
    throw PWDPyException;
  return rslt;
}

PWOBase PWOCallable::call(PWOTuple &args)const {
  PyObject *rslt = PyEval_CallObjectWithKeywords(*this, args, NULL);
  if (rslt == 0)
    throw PWDPyException;
  return rslt;
}

PWOBase PWOCallable::call(PWOTuple &args, PWOMapping &kws)const {
  PyObject *rslt = PyEval_CallObjectWithKeywords(*this, args, kws);
  if (rslt == 0)
    throw PWDPyException;
  return rslt;
}

void Fail(PyObject *exc, const char *msg) {
  PyErr_SetString(exc, msg);
  throw PWDPyException;
}

void FailIfPyErr() {
  PyObject *exc = PyErr_Occurred();
  if (exc != NULL)
    throw PWDPyException;
}
