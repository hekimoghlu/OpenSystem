/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 6, 2021.
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
#import "CoreIPCSecureCoding.h"

#if PLATFORM(COCOA)

#import "ArgumentCodersCocoa.h"
#import "AuxiliaryProcessCreationParameters.h"
#import "WKCrashReporter.h"
#import <wtf/RuntimeApplicationChecks.h>
#import <wtf/TZoneMallocInlines.h>
#import <wtf/text/StringHash.h>

namespace WebKit {

namespace SecureCoding {

static std::unique_ptr<HashSet<String>>& internalClassNamesExemptFromSecureCodingCrash()
{
    static dispatch_once_t onceToken;
    static NeverDestroyed<std::unique_ptr<HashSet<String>>> exemptClassNames;
    dispatch_once(&onceToken, ^{
        if (isInAuxiliaryProcess())
            return;

        NSArray *array = [[NSUserDefaults standardUserDefaults] arrayForKey:@"WebKitCrashOnSecureCodingWithExemptClassesKey"];
        if (!array)
            return;

        exemptClassNames.get() = WTF::makeUnique<HashSet<String>>();

        for (id value in array) {
            if (![value isKindOfClass:[NSString class]])
                continue;
            exemptClassNames.get()->add((NSString *)value);
        }
    });

    return exemptClassNames.get();
}

const HashSet<String>* classNamesExemptFromSecureCodingCrash()
{
    return internalClassNamesExemptFromSecureCodingCrash().get();
}

void applyProcessCreationParameters(const AuxiliaryProcessCreationParameters& parameters)
{
    RELEASE_ASSERT(isInAuxiliaryProcess());

    auto& exemptClassNames = internalClassNamesExemptFromSecureCodingCrash();
    RELEASE_ASSERT(!exemptClassNames);

    if (parameters.classNamesExemptFromSecureCodingCrash)
        *exemptClassNames = WTFMove(*parameters.classNamesExemptFromSecureCodingCrash);
}

} // namespace SecureCoding

#if !HAVE(WK_SECURE_CODING_NSURLREQUEST)
WTF_MAKE_TZONE_ALLOCATED_IMPL(CoreIPCSecureCoding);
#endif

bool conformsToWebKitSecureCoding(id object)
{
    return [object respondsToSelector:@selector(_webKitPropertyListData)]
        && [object respondsToSelector:@selector(_initWithWebKitPropertyListData:)];
}

#if !HAVE(WK_SECURE_CODING_NSURLREQUEST)
NO_RETURN static void crashWithClassName(Class objectClass)
{
    WebKit::logAndSetCrashLogMessage("NSSecureCoding path used for unexpected object"_s);

    std::array<uint64_t, 6> values { 0, 0, 0, 0, 0, 0 };
    strncpy(reinterpret_cast<char*>(values.data()), NSStringFromClass(objectClass).UTF8String, sizeof(values));
    CRASH_WITH_INFO(values[0], values[1], values[2], values[3], values[4], values[5]);
}

CoreIPCSecureCoding::CoreIPCSecureCoding(id object)
    : m_secureCoding((NSObject<NSSecureCoding> *)object)
{
    RELEASE_ASSERT(!m_secureCoding || [object conformsToProtocol:@protocol(NSSecureCoding)]);

    auto* exemptClassNames = SecureCoding::classNamesExemptFromSecureCodingCrash();
    if (!exemptClassNames)
        return;

    if (exemptClassNames->contains(NSStringFromClass([object class])))
        return;

    crashWithClassName([object class]);
}
#endif

} // namespace WebKit

#endif // PLATFORM(COCOA)
