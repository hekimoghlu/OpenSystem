/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 13, 2022.
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

#ifndef _NPY_SCALARTYPES_H_
#define _NPY_SCALARTYPES_H_

/* Internal look-up tables */
#ifdef NPY_ENABLE_SEPARATE_COMPILATION
extern NPY_NO_EXPORT unsigned char
_npy_can_cast_safely_table[NPY_NTYPES][NPY_NTYPES];
extern NPY_NO_EXPORT signed char
_npy_scalar_kinds_table[NPY_NTYPES];
extern NPY_NO_EXPORT signed char
_npy_type_promotion_table[NPY_NTYPES][NPY_NTYPES];
extern NPY_NO_EXPORT signed char
_npy_smallest_type_of_kind_table[NPY_NSCALARKINDS];
extern NPY_NO_EXPORT signed char
_npy_next_larger_type_table[NPY_NTYPES];
#else
NPY_NO_EXPORT unsigned char
_npy_can_cast_safely_table[NPY_NTYPES][NPY_NTYPES];
NPY_NO_EXPORT signed char
_npy_scalar_kinds_table[NPY_NTYPES];
NPY_NO_EXPORT signed char
_npy_type_promotion_table[NPY_NTYPES][NPY_NTYPES];
NPY_NO_EXPORT signed char
_npy_smallest_type_of_kind_table[NPY_NSCALARKINDS];
NPY_NO_EXPORT signed char
_npy_next_larger_type_table[NPY_NTYPES];
#endif

NPY_NO_EXPORT void
initialize_casting_tables(void);

NPY_NO_EXPORT void
initialize_numeric_types(void);

NPY_NO_EXPORT void
format_longdouble(char *buf, size_t buflen, npy_longdouble val, unsigned int prec);

#if PY_VERSION_HEX >= 0x03000000
NPY_NO_EXPORT void
gentype_struct_free(PyObject *ptr);
#else
NPY_NO_EXPORT void
gentype_struct_free(void *ptr, void *arg);
#endif

NPY_NO_EXPORT int
_typenum_fromtypeobj(PyObject *type, int user);

NPY_NO_EXPORT void *
scalar_value(PyObject *scalar, PyArray_Descr *descr);

#endif
