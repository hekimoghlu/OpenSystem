/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 7, 2024.
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
*******************************************************************************
*   Copyright (C) 2011, International Business Machines
*   Corporation and others.  All Rights Reserved.
*******************************************************************************
*   file name:  messageimpl.h
*   encoding:   UTF-8
*   tab size:   8 (not used)
*   indentation:4
*
*   created on: 2011apr04
*   created by: Markus W. Scherer
*/

#ifndef __MESSAGEIMPL_H__
#define __MESSAGEIMPL_H__

#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

#include "unicode/messagepattern.h"

U_NAMESPACE_BEGIN

/**
 * Helper functions for use of MessagePattern.
 * In Java, these are package-private methods in MessagePattern itself.
 * In C++, they are declared here and implemented in messagepattern.cpp.
 */
class U_COMMON_API MessageImpl {
public:
    /**
     * @return true if getApostropheMode()==UMSGPAT_APOS_DOUBLE_REQUIRED
     */
    static UBool jdkAposMode(const MessagePattern &msgPattern) {
        return msgPattern.getApostropheMode()==UMSGPAT_APOS_DOUBLE_REQUIRED;
    }

    /**
     * Appends the s[start, limit[ substring to sb, but with only half of the apostrophes
     * according to JDK pattern behavior.
     */
    static void appendReducedApostrophes(const UnicodeString &s, int32_t start, int32_t limit,
                                         UnicodeString &sb);

    /**
     * Appends the sub-message to the result string.
     * Omits SKIP_SYNTAX and appends whole arguments using appendReducedApostrophes().
     */
    static UnicodeString &appendSubMessageWithoutSkipSyntax(const MessagePattern &msgPattern,
                                                            int32_t msgStart,
                                                            UnicodeString &result);

private:
    MessageImpl() = delete;  // no constructor: all static methods
};

U_NAMESPACE_END

#endif  // !UCONFIG_NO_FORMATTING

#endif  // __MESSAGEIMPL_H__
