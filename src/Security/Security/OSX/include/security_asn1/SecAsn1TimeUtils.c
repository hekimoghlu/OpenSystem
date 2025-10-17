/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 13, 2025.
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
#include <CoreFoundation/CoreFoundation.h>
#include <Security/SecCertificatePriv.h>
#include <Security/X509Templates.h>

#include <utilities/SecCFRelease.h>

#include "plarenas.h"
#include "seccomon.h"
#include "secasn1.h"

#include "SecAsn1TimeUtils.h"

OSStatus SecAsn1DecodeTime(const SecAsn1Item* time, CFAbsoluteTime* date)
{
    CFErrorRef error = NULL;
    PLArenaPool* tmppoolp = NULL;
    OSStatus status = errSecSuccess;

    tmppoolp = PORT_NewArena(1024);
    if (tmppoolp == NULL) {
        return errSecAllocate;
    }

    NSS_Time timeStr;
    if ((status = SEC_ASN1DecodeItem(tmppoolp, &timeStr, kSecAsn1TimeTemplate, time)) !=
        errSecSuccess) {
        goto errOut;
    }

    CFAbsoluteTime result = SecAbsoluteTimeFromDateContentWithError(timeStr.tag, timeStr.item.Data, timeStr.item.Length, &error);
    if (error) {
        status = (OSStatus)CFErrorGetCode(error);
        CFReleaseNull(error);
        goto errOut;
    }

    if (date) {
        *date = result;
    }

errOut:
    if (tmppoolp) {
        PORT_FreeArena(tmppoolp, PR_FALSE);
    }
    return status;
}

static CFStringRef _SecAsn1CreateDateString(CFAbsoluteTime date) {
    // Prefer CFDateFormatter when it's available, as it's the highest fidelity answer.
    CFDateFormatterRef dateFormatter = CFDateFormatterCreateISO8601Formatter(NULL, 0);
    if (dateFormatter) {
        CFStringRef dateString = NULL;
        CFTimeZoneRef timeZone = CFTimeZoneCreateWithTimeIntervalFromGMT(NULL, 0);

        CFDateFormatterSetProperty(dateFormatter, kCFDateFormatterTimeZone, timeZone);
        CFDateFormatterSetFormat(dateFormatter, CFSTR("yyyyMMddHHmmss'Z'"));
        dateString = CFDateFormatterCreateStringWithAbsoluteTime(NULL, dateFormatter, date);

        CFRelease(timeZone);
        CFRelease(dateFormatter);
        return dateString;
    }

    // Fall back to parsing a POSIX timestamp using libSystem.
    const time_t timestamp = date + kCFAbsoluteTimeIntervalSince1970;
    struct tm parsed = {};
    if (gmtime_r(&timestamp, &parsed) == &parsed) {
        return CFStringCreateWithFormat(
            NULL, NULL, CFSTR("%04d%02d%02d%02d%02d%02dZ"),
            parsed.tm_year + 1900, parsed.tm_mon + 1, parsed.tm_mday,
            parsed.tm_hour, parsed.tm_min, parsed.tm_sec);
    };

    return NULL;
}

OSStatus SecAsn1EncodeTime(PLArenaPool *poolp, CFAbsoluteTime date, NSS_Time* asn1Time) {
    OSStatus result = errSecSuccess;
    CFStringRef dateString = NULL;
    CFStringRef fullString = _SecAsn1CreateDateString(date);
    CFRange shortRange = CFRangeMake(2, CFStringGetLength(fullString) - 2);
    if (!fullString) {
        result = errSecAllocate;
        goto errOut;
    }

    if (date < -1609459200.0 || //19500101000000Z
        date > 1546300799.0) {  //20491231235959Z
        // Format: "yyyyMMddHHmmss'Z'"
        dateString = CFRetain(fullString);
        asn1Time->tag = SEC_ASN1_GENERALIZED_TIME;
    } else {
        // Format: "yyMMddHHmmss'Z'", so discard the leading year digits.
        dateString = CFStringCreateWithSubstring(NULL, fullString, shortRange);
        asn1Time->tag = SEC_ASN1_UTC_TIME;
    }

    CFIndex stringLen = CFStringGetLength(dateString);
    if (stringLen < 0) {
        result = errSecAllocate;
        goto errOut;
    }
    asn1Time->item.Length = (size_t)stringLen;
    asn1Time->item.Data = PORT_ArenaAlloc(poolp, (size_t)stringLen);
    if (!asn1Time->item.Data) {
        result = errSecAllocate;
        goto errOut;
    }

    if (stringLen != CFStringGetBytes(dateString, CFRangeMake(0, stringLen), kCFStringEncodingUTF8, 0, false, asn1Time->item.Data, stringLen, NULL)) {
        result = errSecAllocate;
        goto errOut;
    }

errOut:
    CFReleaseNull(fullString);
    CFReleaseNull(dateString);
    return result;
}
