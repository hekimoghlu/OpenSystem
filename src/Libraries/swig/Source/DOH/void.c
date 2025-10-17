/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 9, 2023.
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
char cvsroot_void_c[] = "$Id: void.c 9607 2006-12-05 22:11:40Z beazley $";

#include "dohint.h"

typedef struct {
  void *ptr;
  void (*del) (void *);
} VoidObj;

/* -----------------------------------------------------------------------------
 * Void_delete()
 *
 * Delete a void object. Invokes the destructor supplied at the time of creation.
 * ----------------------------------------------------------------------------- */

static void Void_delete(DOH *vo) {
  VoidObj *v = (VoidObj *) ObjData(vo);
  if (v->del)
    (*v->del) (v->ptr);
  DohFree(v);
}

/* -----------------------------------------------------------------------------
 * Void_copy()
 *
 * Copies a void object.  This is only a shallow copy. The object destruction
 * function is not copied in order to avoid potential double-free problems.
 * ----------------------------------------------------------------------------- */

static DOH *Void_copy(DOH *vo) {
  VoidObj *v = (VoidObj *) ObjData(vo);
  return NewVoid(v->ptr, 0);
}

/* -----------------------------------------------------------------------------
 * Void_data()
 *
 * Returns the void * stored in the object.
 * ----------------------------------------------------------------------------- */

static void *Void_data(DOH *vo) {
  VoidObj *v = (VoidObj *) ObjData(vo);
  return v->ptr;
}

static DohObjInfo DohVoidType = {
  "VoidObj",			/* objname */
  Void_delete,			/* doh_del */
  Void_copy,			/* doh_copy */
  0,				/* doh_clear */
  0,				/* doh_str */
  Void_data,			/* doh_data */
  0,				/* doh_dump */
  0,				/* doh_len */
  0,				/* doh_hash    */
  0,				/* doh_cmp */
  0,				/* doh_equal    */
  0,				/* doh_first    */
  0,				/* doh_next     */
  0,				/* doh_setfile */
  0,				/* doh_getfile */
  0,				/* doh_setline */
  0,				/* doh_getline */
  0,				/* doh_mapping */
  0,				/* doh_sequence */
  0,				/* doh_file  */
  0,				/* doh_string */
  0,				/* doh_reserved */
  0,				/* clientdata */
};

/* -----------------------------------------------------------------------------
 * NewVoid()
 *
 * Creates a new Void object given a void * and an optional destructor function.
 * ----------------------------------------------------------------------------- */

DOH *DohNewVoid(void *obj, void (*del) (void *)) {
  VoidObj *v;
  v = (VoidObj *) DohMalloc(sizeof(VoidObj));
  v->ptr = obj;
  v->del = del;
  return DohObjMalloc(&DohVoidType, v);
}
