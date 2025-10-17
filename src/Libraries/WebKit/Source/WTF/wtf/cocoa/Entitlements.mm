/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 12, 2024.
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
#import <wtf/cocoa/Entitlements.h>

#import <wtf/OSObjectPtr.h>
#import <wtf/RetainPtr.h>
#import <wtf/cf/TypeCastsCF.h>
#import <wtf/text/WTFString.h>

namespace WTF {

bool hasEntitlement(SecTaskRef task, ASCIILiteral entitlement)
{
    if (!task)
        return false;
    auto string = entitlement.createCFString();
    return adoptCF(SecTaskCopyValueForEntitlement(task, string.get(), nullptr)) == kCFBooleanTrue;
}

bool hasEntitlement(audit_token_t token, ASCIILiteral entitlement)
{
    return hasEntitlement(adoptCF(SecTaskCreateWithAuditToken(kCFAllocatorDefault, token)).get(), entitlement);
}

bool hasEntitlement(xpc_connection_t connection, StringView entitlement)
{
    auto value = adoptOSObject(xpc_connection_copy_entitlement_value(connection, entitlement.utf8().data()));
    return value && xpc_get_type(value.get()) == XPC_TYPE_BOOL && xpc_bool_get_value(value.get());
}

bool hasEntitlement(xpc_connection_t connection, ASCIILiteral entitlement)
{
    auto value = adoptOSObject(xpc_connection_copy_entitlement_value(connection, entitlement.characters()));
    return value && xpc_get_type(value.get()) == XPC_TYPE_BOOL && xpc_bool_get_value(value.get());
}

bool processHasEntitlement(ASCIILiteral entitlement)
{
    return hasEntitlement(adoptCF(SecTaskCreateFromSelf(kCFAllocatorDefault)).get(), entitlement);
}

bool hasEntitlementValue(audit_token_t token, ASCIILiteral entitlement, ASCIILiteral value)
{
    auto secTaskForToken = adoptCF(SecTaskCreateWithAuditToken(kCFAllocatorDefault, token));
    if (!secTaskForToken)
        return false;

    auto string = entitlement.createCFString();
    String entitlementValue = dynamic_cf_cast<CFStringRef>(adoptCF(SecTaskCopyValueForEntitlement(secTaskForToken.get(), string.get(), nullptr)).get());
    return entitlementValue == value;
}

bool hasEntitlementValueInArray(audit_token_t token, ASCIILiteral entitlement, ASCIILiteral value)
{
    auto secTaskForToken = adoptCF(SecTaskCreateWithAuditToken(kCFAllocatorDefault, token));
    if (!secTaskForToken)
        return false;

    auto string = entitlement.createCFString();
    auto entitlementValue = adoptCF(SecTaskCopyValueForEntitlement(secTaskForToken.get(), string.get(), nullptr)).get();
    if (!entitlementValue || CFGetTypeID(entitlementValue) != CFArrayGetTypeID())
        return false;

    RetainPtr<CFArrayRef> array = static_cast<CFArrayRef>(entitlementValue);

    for (CFIndex i = 0; i < CFArrayGetCount(array.get()); ++i) {
        auto element = CFArrayGetValueAtIndex(array.get(), i);
        if (CFGetTypeID(element) != CFStringGetTypeID())
            continue;
        CFStringRef stringElement = static_cast<CFStringRef>(element);
        if (value == stringElement)
            return true;
    }

    return false;
}

} // namespace WTF
