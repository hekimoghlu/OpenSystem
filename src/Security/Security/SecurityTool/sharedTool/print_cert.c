/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 12, 2023.
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
#include "print_cert.h"
#include <CoreFoundation/CoreFoundation.h>
#include <Security/SecCertificatePriv.h>
#include <Security/SecTrustPriv.h>
#include <utilities/SecIOFormat.h>
#include <utilities/SecCFWrappers.h>


static void printPlist(CFArrayRef plist, CFIndex indent, CFIndex maxWidth) {
    CFIndex count = CFArrayGetCount(plist);
    CFIndex ix;
    for (ix = 0; ix < count ; ++ix) {
        CFDictionaryRef prop = (CFDictionaryRef)CFArrayGetValueAtIndex(plist,
            ix);
        CFStringRef pType = (CFStringRef)CFDictionaryGetValue(prop,
            kSecPropertyKeyType);
        CFStringRef label = (CFStringRef)CFDictionaryGetValue(prop,
            kSecPropertyKeyLabel);
        CFStringRef llabel = (CFStringRef)CFDictionaryGetValue(prop,
            kSecPropertyKeyLocalizedLabel);
        CFTypeRef value = (CFTypeRef)CFDictionaryGetValue(prop,
            kSecPropertyKeyValue);

        bool isSection = CFEqual(pType, kSecPropertyTypeSection);
        CFMutableStringRef line = CFStringCreateMutable(NULL, 0);
        CFIndex jx = 0;
        for (jx = 0; jx < indent; ++jx) {
            CFStringAppend(line, CFSTR("    "));
        }
        if (llabel) {
            CFStringAppend(line, llabel);
            if (!isSection) {
                for (jx = CFStringGetLength(llabel) + indent * 4;
                    jx < maxWidth; ++jx) {
                    CFStringAppend(line, CFSTR(" "));
                }
                CFStringAppend(line, CFSTR(" : "));
            }
        }
        if (CFEqual(pType, kSecPropertyTypeWarning)) {
            CFStringAppend(line, CFSTR("*WARNING* "));
            CFStringAppend(line, (CFStringRef)value);
        } else if (CFEqual(pType, kSecPropertyTypeError)) {
            CFStringAppend(line, CFSTR("*ERROR* "));
            CFStringAppend(line, (CFStringRef)value);
        } else if (CFEqual(pType, kSecPropertyTypeSuccess)) {
            CFStringAppend(line, CFSTR("*OK* "));
            CFStringAppend(line, (CFStringRef)value);
        } else if (CFEqual(pType, kSecPropertyTypeTitle)) {
            CFStringAppend(line, CFSTR("*"));
            CFStringAppend(line, (CFStringRef)value);
            CFStringAppend(line, CFSTR("*"));
        } else if (CFEqual(pType, kSecPropertyTypeSection)) {
        } else if (CFEqual(pType, kSecPropertyTypeData)) {
            CFDataRef data = (CFDataRef)value;
            CFIndex length = CFDataGetLength(data);
            if (length > 20)
                CFStringAppendFormat(line, NULL, CFSTR("[%" PRIdCFIndex " bytes] "), length);
            const UInt8 *bytes = CFDataGetBytePtr(data);
            for (jx = 0; jx < length; ++jx) {
                if (jx == 0)
                    CFStringAppendFormat(line, NULL, CFSTR("%02X"), bytes[jx]);
                else if (jx < 15 || length <= 20)
                    CFStringAppendFormat(line, NULL, CFSTR(" %02X"),
                        bytes[jx]);
                else {
                    CFStringAppend(line, CFSTR(" ..."));
                    break;
                }
            }
        } else if (CFEqual(pType, kSecPropertyTypeString)) {
            CFStringAppend(line, (CFStringRef)value);
        } else if (CFEqual(pType, kSecPropertyTypeDate)) {
            CFDateRef date = (CFDateRef)value;
            CFLocaleRef lc = CFLocaleCopyCurrent();
            CFDateFormatterRef df = CFDateFormatterCreate(NULL, lc, kCFDateFormatterMediumStyle, kCFDateFormatterLongStyle);
            CFStringRef ds;
            if (df) {
                CFTimeZoneRef tz = CFTimeZoneCreateWithTimeIntervalFromGMT(NULL, 0.0);
                CFDateFormatterSetProperty(df, kCFDateFormatterTimeZone, tz);
                CFRelease(tz);
                ds = CFDateFormatterCreateStringWithDate(NULL, df, date);
                CFRelease(df);
            } else {
                ds = CFStringCreateWithFormat(NULL, NULL, CFSTR("%g"), CFDateGetAbsoluteTime(date));
            }
            CFStringAppend(line, ds);
            CFRelease(ds);
            CFRelease(lc);
        } else if (CFEqual(pType, kSecPropertyTypeURL)) {
            CFURLRef url = (CFURLRef)value;
            CFStringAppend(line, CFSTR("<"));
            CFStringAppend(line, CFURLGetString(url));
            CFStringAppend(line, CFSTR(">"));
        } else {
            CFStringAppendFormat(line, NULL, CFSTR("*unknown type %@* = %@"),
            pType, value);
        }

		if (!isSection || label)
			CFStringWriteToFileWithNewline(line, stdout);
		CFRelease(line);
        if (isSection) {
            printPlist((CFArrayRef)value, indent + 1, maxWidth);
        }
    }
}

static CFIndex maxLabelWidth(CFArrayRef plist, CFIndex indent) {
    CFIndex count = CFArrayGetCount(plist);
    CFIndex ix;
    CFIndex maxWidth = 0;
    for (ix = 0; ix < count ; ++ix) {
        CFDictionaryRef prop = (CFDictionaryRef)CFArrayGetValueAtIndex(plist,
            ix);
        CFStringRef pType = (CFStringRef)CFDictionaryGetValue(prop,
            kSecPropertyKeyType);
        CFStringRef llabel = (CFStringRef)CFDictionaryGetValue(prop,
            kSecPropertyKeyLocalizedLabel);
        CFTypeRef value = (CFTypeRef)CFDictionaryGetValue(prop,
            kSecPropertyKeyValue);

        if (CFEqual(pType, kSecPropertyTypeSection)) {
            CFIndex width = maxLabelWidth((CFArrayRef)value, indent + 1);
            if (width > maxWidth)
                maxWidth = width;
        } else if (llabel) {
            CFIndex width = indent * 4 + CFStringGetLength(llabel);
            if (width > maxWidth)
                maxWidth = width;
        }
    }

    return maxWidth;
}

void print_plist(CFArrayRef plist) {
    if (plist)
        printPlist(plist, 0, maxLabelWidth(plist, 0));
    else
        printf("NULL plist\n");
}

void print_cert(SecCertificateRef cert, bool verbose) {
    CFArrayRef plist;
    if (verbose)
        plist = SecCertificateCopyProperties(cert);
    else {
        CFAbsoluteTime now = CFAbsoluteTimeGetCurrent();
        plist = SecCertificateCopySummaryProperties(cert, now);
    }

    CFStringRef subject = SecCertificateCopySubjectString(cert);
    if (subject) {
        CFStringWriteToFileWithNewline(subject, stdout);
        CFRelease(subject);
    } else {
        CFStringWriteToFileWithNewline(CFSTR("no subject"), stdout);
    }

    print_plist(plist);
    CFReleaseSafe(plist);
}
