/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 24, 2023.
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

#if !defined(TUPLE_H_INCLUDED_)
#define TUPLE_H_INCLUDED_

#include "sequence.h"
#include <string>

namespace py {

class tuple : public sequence
{ 
public:

  //-------------------------------------------------------------------------
  // constructors
  //-------------------------------------------------------------------------
  tuple(int sz=0) : sequence (PyTuple_New(sz))  { lose_ref(_obj); }
  tuple(const tuple& other) : sequence(other) { }
  tuple(PyObject* obj) : sequence(obj) { _violentTypeCheck(); }
  tuple(const list& lst)
    : sequence (PyList_AsTuple(lst)) { lose_ref(_obj); }
    
  //-------------------------------------------------------------------------
  // destructor
  //-------------------------------------------------------------------------    
  virtual ~tuple() {};

  //-------------------------------------------------------------------------
  // operator=
  //-------------------------------------------------------------------------
  virtual tuple& operator=(const tuple& other) {
    grab_ref(other);
    return *this;
  };
  /*virtual*/ tuple& operator=(const object& other) {
    grab_ref(other);
    _violentTypeCheck();
    return *this;
  };

  //-------------------------------------------------------------------------
  // type checking
  //-------------------------------------------------------------------------    
  virtual void _violentTypeCheck() {
    if (!PyTuple_Check(_obj)) {
      grab_ref(0);
      fail(PyExc_TypeError, "Not a Python Tuple");
    }
  };

  //-------------------------------------------------------------------------
  // set_item
  //
  // We have to do a little extra checking here, because tuples can only
  // be assigned to if there is only a single reference to them.
  //-------------------------------------------------------------------------      
  virtual void set_item(int ndx, object& val) {
    if (_obj->ob_refcnt != 1)
      fail(PyExc_TypeError,"Tuples values can't be set if ref count > 1\n");  
    int rslt = PyTuple_SetItem(_obj, ndx, val);
    val.disown(); //when using PyTuple_SetItem, he steals my reference
    if (rslt==-1)
      throw 1;
  };
  
  //-------------------------------------------------------------------------
  // operator[] -- const and non-const versions of element access.
  //-------------------------------------------------------------------------    
  indexed_ref operator [] (int i) {   
    // get a "borrowed" refcount    
    PyObject* o = PyTuple_GetItem(_obj, i);  
    // don't throw error for when [] fails because it might be on left hand 
    // side (a[0] = 1).  If the tuple was just created, it will be filled 
    // with NULL values, and setting the values should be ok.  However, we
    // do want to catch index errors that might occur on the right hand side
    // (obj = a[4] when a has len==3).
    if (!o) {
      if (PyErr_ExceptionMatches(PyExc_IndexError))
        throw 1;
    }
    return indexed_ref(o, *this, i); // this increfs
  };
};// class tuple

} // namespace

#endif // TUPLE_H_INCLUDED_
