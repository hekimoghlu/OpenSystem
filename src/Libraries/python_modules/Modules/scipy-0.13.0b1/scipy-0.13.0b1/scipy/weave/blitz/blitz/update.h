/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 26, 2025.
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

// -*- C++ -*-
/***************************************************************************
 * blitz/update.h      Declaration of the _bz_XXXX updater classes
 *
 * $Id: update.h 1414 2005-11-01 22:04:59Z cookedm $
 *
 * Copyright (C) 1997-2001 Todd Veldhuizen <tveldhui@oonumerics.org>
 *
 * This code was relicensed under the modified BSD license for use in SciPy
 * by Todd Veldhuizen (see LICENSE.txt in the weave directory).
 *
 *
 * Suggestions:          blitz-dev@oonumerics.org
 * Bugs:                 blitz-bugs@oonumerics.org
 *
 * For more information, please see the Blitz++ Home Page:
 *    http://oonumerics.org/blitz/
 *
 ***************************************************************************/

#ifndef BZ_UPDATE_H
#define BZ_UPDATE_H

#include <blitz/blitz.h>

BZ_NAMESPACE(blitz)

class _bz_updater_base { };

#define BZ_DECL_UPDATER(name,op,symbol)                     \
  template<typename X, typename Y>                          \
  class name : public _bz_updater_base {                    \
  public:                                                   \
    static inline void update(X& restrict x, Y y)           \
    { x op y; }                                             \
    static void prettyPrint(BZ_STD_SCOPE(string) &str)      \
    { str += symbol; }                                      \
  }

template<typename X, typename Y>
class _bz_update : public _bz_updater_base {
  public:
    static inline void update(X& restrict x, Y y)
    { x = (X)y; }

    static void prettyPrint(BZ_STD_SCOPE(string) &str)
    { str += "="; }
};

BZ_DECL_UPDATER(_bz_plus_update, +=, "+=");
BZ_DECL_UPDATER(_bz_minus_update, -=, "-=");
BZ_DECL_UPDATER(_bz_multiply_update, *=, "*=");
BZ_DECL_UPDATER(_bz_divide_update, /=, "/=");
BZ_DECL_UPDATER(_bz_mod_update, %=, "%=");
BZ_DECL_UPDATER(_bz_xor_update, ^=, "^=");
BZ_DECL_UPDATER(_bz_bitand_update, &=, "&=");
BZ_DECL_UPDATER(_bz_bitor_update, |=, "|=");
BZ_DECL_UPDATER(_bz_shiftl_update, <<=, "<<=");
BZ_DECL_UPDATER(_bz_shiftr_update, >>=, ">>=");

BZ_NAMESPACE_END

#endif // BZ_UPDATE_H

