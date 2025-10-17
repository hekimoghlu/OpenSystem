/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 4, 2023.
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
#import "NetworkRTCUtilitiesCocoa.h"

#if USE(LIBWEBRTC)

#import <WebCore/RegistrableDomain.h>
#import <wtf/SoftLinking.h>
#import <wtf/text/WTFString.h>

SOFT_LINK_LIBRARY(libnetworkextension)
SOFT_LINK_CLASS(libnetworkextension, NEHelperTrackerDisposition_t)
SOFT_LINK_CLASS(libnetworkextension, NEHelperTrackerAppInfoRef)
SOFT_LINK_CLASS(libnetworkextension, NEHelperTrackerDomainContextRef)
SOFT_LINK(libnetworkextension, NEHelperTrackerGetDisposition, NEHelperTrackerDisposition_t*, (NEHelperTrackerAppInfoRef *app_info_ref, CFArrayRef domains, NEHelperTrackerDomainContextRef *trackerDomainContextRef, CFIndex *trackerDomainIndex), (app_info_ref, domains, trackerDomainContextRef, trackerDomainIndex))

SOFT_LINK_LIBRARY_OPTIONAL(libnetwork)
SOFT_LINK_OPTIONAL(libnetwork, nw_parameters_set_attributed_bundle_identifier, void, __cdecl, (nw_parameters_t, const char*))

namespace WebKit {

void setNWParametersApplicationIdentifiers(nw_parameters_t parameters, const char* sourceApplicationBundleIdentifier, std::optional<audit_token_t> sourceApplicationAuditToken, const String& attributedBundleIdentifier)
{
    if (sourceApplicationBundleIdentifier && *sourceApplicationBundleIdentifier)
        nw_parameters_set_source_application_by_bundle_id(parameters, sourceApplicationBundleIdentifier);
    else if (sourceApplicationAuditToken)
        nw_parameters_set_source_application(parameters, *sourceApplicationAuditToken);

    if (!attributedBundleIdentifier.isEmpty() && nw_parameters_set_attributed_bundle_identifierPtr())
        nw_parameters_set_attributed_bundle_identifierPtr()(parameters, attributedBundleIdentifier.utf8().data());
}

void setNWParametersTrackerOptions(nw_parameters_t parameters, bool shouldBypassRelay, bool isFirstParty, bool isKnownTracker)
{
    if (shouldBypassRelay)
        nw_parameters_set_account_id(parameters, "com.apple.safari.peertopeer");
    nw_parameters_set_is_third_party_web_content(parameters, !isFirstParty);
    nw_parameters_set_is_known_tracker(parameters, isKnownTracker);
}

bool isKnownTracker(const WebCore::RegistrableDomain& domain)
{
    NSArray<NSString *> *domains = @[domain.string()];
    NEHelperTrackerDomainContextRef *context = nil;
    CFIndex index = 0;
    return !!NEHelperTrackerGetDisposition(nullptr, (CFArrayRef)domains, context, &index);
}

std::optional<uint32_t> trafficClassFromDSCP(rtc::DiffServCodePoint dscpValue)
{
    switch (dscpValue) {
    case rtc::DiffServCodePoint::DSCP_NO_CHANGE:
        return { };
    case rtc::DiffServCodePoint::DSCP_CS0:
        return SO_TC_BE;
    case rtc::DiffServCodePoint::DSCP_CS1:
        return SO_TC_BK_SYS;
    case rtc::DiffServCodePoint::DSCP_AF41:
        return SO_TC_VI;
    case rtc::DiffServCodePoint::DSCP_AF42:
        return SO_TC_VI;
    case rtc::DiffServCodePoint::DSCP_EF:
        return SO_TC_VO;
    case rtc::DiffServCodePoint::DSCP_AF11:
    case rtc::DiffServCodePoint::DSCP_AF12:
    case rtc::DiffServCodePoint::DSCP_AF13:
    case rtc::DiffServCodePoint::DSCP_CS2:
    case rtc::DiffServCodePoint::DSCP_AF21:
    case rtc::DiffServCodePoint::DSCP_AF22:
    case rtc::DiffServCodePoint::DSCP_AF23:
    case rtc::DiffServCodePoint::DSCP_CS3:
    case rtc::DiffServCodePoint::DSCP_AF31:
    case rtc::DiffServCodePoint::DSCP_AF32:
    case rtc::DiffServCodePoint::DSCP_AF33:
    case rtc::DiffServCodePoint::DSCP_CS4:
    case rtc::DiffServCodePoint::DSCP_AF43:
    case rtc::DiffServCodePoint::DSCP_CS5:
    case rtc::DiffServCodePoint::DSCP_CS6:
    case rtc::DiffServCodePoint::DSCP_CS7:
        break;
    };
    return { };
}

} // namespace WebKit

#endif // USE(LIBWEBRTC)
