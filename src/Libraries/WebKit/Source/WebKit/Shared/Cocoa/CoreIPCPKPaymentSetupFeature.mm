/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 5, 2023.
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
#import "CoreIPCPKPaymentSetupFeature.h"

#if USE(PASSKIT)

#import <wtf/RuntimeApplicationChecks.h>
#import <wtf/cocoa/VectorCocoa.h>

#import <pal/cocoa/PassKitSoftLink.h>

namespace WebKit {

CoreIPCPKPaymentSetupFeature::CoreIPCPKPaymentSetupFeature(PKPaymentSetupFeature *feature)
    : m_data(makeVector([NSKeyedArchiver archivedDataWithRootObject:feature requiringSecureCoding:YES error:nil])) { }

RetainPtr<id> CoreIPCPKPaymentSetupFeature::toID() const
{
    RetainPtr data = adoptNS([[NSData alloc] initWithBytesNoCopy:const_cast<uint8_t*>(m_data.data()) length:m_data.size() freeWhenDone:NO]);
    RELEASE_ASSERT(isInWebProcess());
    return [NSKeyedUnarchiver unarchivedObjectOfClass:PAL::getPKPaymentSetupFeatureClass() fromData:data.get() error:nil];
}

} // namespace WebKit

#endif // USE(PASSKIT)
