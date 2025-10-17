/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 9, 2022.
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
#import "config.h"
#import "LegacyWebArchive.h"

namespace WebCore {

static NSString * const LegacyWebArchiveResourceResponseKey = @"WebResourceResponse";

// FIXME: If is is possible to parse in a serialized NSURLResponse manually, without using
// NSKeyedUnarchiver, manipulating plists directly, we would prefer to do that instead.
ResourceResponse LegacyWebArchive::createResourceResponseFromMacArchivedData(CFDataRef responseData)
{    
    ASSERT(responseData);
    if (!responseData)
        return ResourceResponse();
    
    NSURLResponse *response = nil;
    auto unarchiver = adoptNS([[NSKeyedUnarchiver alloc] initForReadingFromData:(__bridge NSData *)responseData error:nullptr]);
    unarchiver.get().decodingFailurePolicy = NSDecodingFailurePolicyRaiseException;
    @try {
        response = [unarchiver decodeObjectOfClass:NSURLResponse.class forKey:LegacyWebArchiveResourceResponseKey];
        [unarchiver finishDecoding];
    } @catch (NSException *exception) {
        LOG_ERROR("Failed to decode NS(HTTP)URLResponse: %@", exception);
        response = nil;
    }

    return ResourceResponse(response);
}

RetainPtr<CFDataRef> LegacyWebArchive::createPropertyListRepresentation(const ResourceResponse& response)
{    
    NSURLResponse *nsResponse = response.nsURLResponse();
    ASSERT(nsResponse);
    if (!nsResponse)
        return nullptr;

    auto archiver = adoptNS([[NSKeyedArchiver alloc] initRequiringSecureCoding:YES]);
    [archiver encodeObject:nsResponse forKey:LegacyWebArchiveResourceResponseKey];
    return (__bridge CFDataRef)archiver.get().encodedData;
}

}
