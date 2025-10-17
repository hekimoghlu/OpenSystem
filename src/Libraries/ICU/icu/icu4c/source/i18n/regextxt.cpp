/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 19, 2022.
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
/********************************************************************
 * COPYRIGHT:
 * Copyright (c) 2008-2011, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/
//
//  file:  regextxt.cpp
//
//  This file contains utility code for supporting UText in the regular expression engine.
//

#include "unicode/utf.h"
#include "regextxt.h"

U_NAMESPACE_BEGIN

U_CFUNC char16_t U_CALLCONV
uregex_utext_unescape_charAt(int32_t offset, void *ct) {
    struct URegexUTextUnescapeCharContext *context = (struct URegexUTextUnescapeCharContext *)ct;
    UChar32 c;
    if (offset == context->lastOffset + 1) {
        c = UTEXT_NEXT32(context->text);
        context->lastOffset++;
    } else if (offset == context->lastOffset) {
        c = UTEXT_PREVIOUS32(context->text);
        UTEXT_NEXT32(context->text);
    } else {
        utext_moveIndex32(context->text, offset - context->lastOffset - 1);
        c = UTEXT_NEXT32(context->text);
        context->lastOffset = offset;
    }

    // !!!: Doesn't handle characters outside BMP
    if (U_IS_BMP(c)) {
        return (char16_t)c;
    } else {
        return 0;
    }
}

U_CFUNC char16_t U_CALLCONV
uregex_ucstr_unescape_charAt(int32_t offset, void *context) {
    return ((char16_t *)context)[offset];
}

U_NAMESPACE_END
