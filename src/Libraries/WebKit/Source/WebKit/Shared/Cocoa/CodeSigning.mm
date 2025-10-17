/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 14, 2025.
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
#import "CodeSigning.h"

#if PLATFORM(COCOA)

#import <wtf/RetainPtr.h>
#import <wtf/spi/cocoa/SecuritySPI.h>
#import <wtf/spi/darwin/CodeSignSPI.h>
#import <wtf/text/WTFString.h>

namespace WebKit {

static String codeSigningIdentifier(SecTaskRef task)
{
    return adoptCF(SecTaskCopySigningIdentifier(task, nullptr)).get();
}

String codeSigningIdentifierForCurrentProcess()
{
    return codeSigningIdentifier(adoptCF(SecTaskCreateFromSelf(kCFAllocatorDefault)).get());
}

String codeSigningIdentifier(xpc_connection_t connection)
{
    auto pair = codeSigningIdentifierAndPlatformBinaryStatus(connection);
    return pair.first;
}

bool currentProcessIsPlatformBinary()
{
    auto task = adoptCF(SecTaskCreateFromSelf(kCFAllocatorDefault));
    return SecTaskGetCodeSignStatus(task.get()) & CS_PLATFORM_BINARY;
}

static std::pair<String, bool> codeSigningIdentifierAndPlatformBinaryStatus(audit_token_t auditToken)
{
    auto task = adoptCF(SecTaskCreateWithAuditToken(kCFAllocatorDefault, auditToken));
    bool isPlatformBinary = SecTaskGetCodeSignStatus(task.get()) & CS_PLATFORM_BINARY;
    auto signingIdentifier = codeSigningIdentifier(task.get());
    return std::make_pair(signingIdentifier, isPlatformBinary);
}

std::pair<String, bool> codeSigningIdentifierAndPlatformBinaryStatus(xpc_connection_t connection)
{
    audit_token_t auditToken;
    xpc_connection_get_audit_token(connection, &auditToken);

    return codeSigningIdentifierAndPlatformBinaryStatus(auditToken);
}

String codeSigningIdentifier(audit_token_t token)
{
    auto pair = codeSigningIdentifierAndPlatformBinaryStatus(token);
    return pair.first;
}

} // namespace WebKit

#endif // PLATFORM(COCOA)
