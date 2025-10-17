/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 15, 2023.
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
#if !defined(STR_H_INCLUDED_)
#define STR_H_INCLUDED_

#include <string>
#include "object.h"
#include "sequence.h"

namespace py {

class str : public sequence
{
public:
  str() : sequence() {};
  str(const char* s)
    : sequence(PyString_FromString((char* )s)) { lose_ref(_obj); }
  str(const char* s, int sz)
    : sequence(PyString_FromStringAndSize((char* )s, sz)) {  lose_ref(_obj); }
  str(const str& other)
    : sequence(other) {};
  str(PyObject* obj)
    : sequence(obj) { _violentTypeCheck(); };
  str(const object& other)
    : sequence(other) { _violentTypeCheck(); };
  virtual ~str() {};

  virtual str& operator=(const str& other) {
    grab_ref(other);
    return *this;
  };
  str& operator=(const object& other) {
    grab_ref(other);
    _violentTypeCheck();
    return *this;
  };
  virtual void _violentTypeCheck() {
    if (!PyString_Check(_obj)) {
      grab_ref(0);
      fail(PyExc_TypeError, "Not a Python String");
    }
  };
  operator const char* () const {
    return PyString_AsString(_obj);
  };
  /*
  static str format(const str& fmt, tuple& args){
    PyObject * rslt =PyString_Format(fmt, args);
    if (rslt==0)
      fail(PyExc_RuntimeError, "string format failed");
    return lose_ref(rslt);
  };
  */
}; // class str

} // namespace

#endif // STR_H_INCLUDED_
