/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 22, 2025.
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
#include <Security/Security.h>
#include <Security/SecBase.h>
#include "SecureDownload.h"


OSStatus SecureDownloadCreateWithTicket (CFDataRef ticket,
										 SecureDownloadTrustSetupCallback setup,
										 void* setupContext,
										 SecureDownloadTrustEvaluateCallback evaluate,
										 void* evaluateContext,
										 SecureDownloadRef* downloadRef)
{
    return errSecUnimplemented;
}



OSStatus SecureDownloadCopyURLs (SecureDownloadRef downloadRef, CFArrayRef* urls)
{
    return errSecUnimplemented;
}



OSStatus SecureDownloadCopyName (SecureDownloadRef downloadRef, CFStringRef* name)
{
    return errSecUnimplemented;
}



OSStatus SecureDownloadCopyCreationDate (SecureDownloadRef downloadRef, CFDateRef* date)
{
    return errSecUnimplemented;
}



OSStatus SecureDownloadGetDownloadSize (SecureDownloadRef downloadRef, SInt64* size)
{
    return errSecUnimplemented;
}



OSStatus SecureDownloadUpdateWithData (SecureDownloadRef downloadRef, CFDataRef data)
{
    return errSecUnimplemented;
}



OSStatus SecureDownloadFinished (SecureDownloadRef downloadRef)
{
    return errSecUnimplemented;
}



OSStatus SecureDownloadRelease (SecureDownloadRef downloadRef)
{
    return errSecUnimplemented;
}



OSStatus SecureDownloadCopyTicketLocation (CFURLRef url, CFURLRef *ticketLocation)
{
    return errSecUnimplemented;
}
