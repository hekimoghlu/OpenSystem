/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 1, 2022.
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
**********************************************************************
*   Copyright (C) 2001-2014 International Business Machines
*   Corporation and others.  All Rights Reserved.
**********************************************************************
*  FILE NAME : ustream.h
*
*   Modification History:
*
*   Date        Name        Description
*   06/25/2001  grhoten     Move iostream from unistr.h
******************************************************************************
*/

#ifndef USTREAM_H
#define USTREAM_H

#include "unicode/utypes.h"

#if U_SHOW_CPLUSPLUS_API

#include "unicode/unistr.h"

#if !UCONFIG_NO_CONVERSION  // not available without conversion

/**
 * \file
 * \brief C++ API: Unicode iostream like API
 *
 * At this time, this API is very limited. It contains
 * operator<< and operator>> for UnicodeString manipulation with the
 * C++ I/O stream API.
 */

#if defined(__GLIBCXX__)
namespace std { class type_info; } // WORKAROUND: http://llvm.org/bugs/show_bug.cgi?id=13364
#endif

#include <iostream>

U_NAMESPACE_BEGIN

/**
 * Write the contents of a UnicodeString to a C++ ostream. This functions writes
 * the characters in a UnicodeString to an ostream. The UChars in the
 * UnicodeString are converted to the char based ostream with the default
 * converter.
 * @stable 3.0
 */
U_IO_API std::ostream & U_EXPORT2 operator<<(std::ostream& stream, const UnicodeString& s);

/**
 * Write the contents from a C++ istream to a UnicodeString. The UChars in the
 * UnicodeString are converted from the char based istream with the default
 * converter.
 * @stable 3.0
 */
U_IO_API std::istream & U_EXPORT2 operator>>(std::istream& stream, UnicodeString& s);
U_NAMESPACE_END

#endif

/* No operator for UChar because it can conflict with wchar_t  */

#endif /* U_SHOW_CPLUSPLUS_API */

#endif
