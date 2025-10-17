/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 20, 2023.
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

// Â© 2016 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html
/*
******************************************************************************
*
*   Copyright (C) 2012-2016, International Business Machines
*   Corporation and others.  All Rights Reserved.
*
******************************************************************************
*/

#ifndef __UTYPEINFO_H__
#define __UTYPEINFO_H__

// Windows header <typeinfo> does not define 'exception' in 'std' namespace.
// Therefore, a project using ICU cannot be compiled with _HAS_EXCEPTIONS
// set to 0 on Windows with Visual Studio. To work around that, we have to
// include <exception> explicitly and add using statement below.
// Whenever 'typeid' is used, this header has to be included
// instead of <typeinfo>.
// Visual Studio 10 emits warning 4275 with this change. If you compile
// with exception disabled, you have to suppress warning 4275.
#if defined(_MSC_VER) && _HAS_EXCEPTIONS == 0
#include <exception>
using std::exception;
#endif
#if defined(__GLIBCXX__)
namespace std { class type_info; } // WORKAROUND: http://llvm.org/bugs/show_bug.cgi?id=13364
#endif
#include <typeinfo>  // for 'typeid' to work

#endif
