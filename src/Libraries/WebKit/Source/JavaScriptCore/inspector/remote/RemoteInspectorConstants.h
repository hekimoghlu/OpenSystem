/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 16, 2023.
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
#pragma once

// WIRConstants are "Web Inspector Relay" constants shared between
// the WebInspector framework on the OS X side, webinspectord, and
// iOS WebKit on the device side.

#define WIRXPCMachPortName                      "com.apple.webinspector"
#define WIRXPCDebuggerServiceName               "com.apple.webinspector.debugger"
#define WIRServiceAvailableNotification         "com.apple.webinspectord.available"
#define WIRServiceAvailabilityCheckNotification "com.apple.webinspectord.availability_check"
#define WIRServiceEnabledNotification           "com.apple.webinspectord.enabled"
#define WIRServiceDisabledNotification          "com.apple.webinspectord.disabled"
#define WIRRemoteAutomationEnabledNotification  "com.apple.webinspectord.remote_automation_enabled"
#define WIRRemoteAutomationDisabledNotification "com.apple.webinspectord.remote_automation_disabled"

// COMPATIBILITY(macOS 13): The key string is intentionally mismatched to support old relays.
#define WIRGlobalNotifyStateName                    "com.apple.webinspectord.automatic_inspection_enabled"
#define WIRGlobalNotifyStateAutomaticInspection     1ULL << 0
#define WIRGlobalNotifyStateSimulateCustomerInstall 1ULL << 63

#define WIRApplicationIdentifierKey             @"WIRApplicationIdentifierKey"
#define WIRApplicationBundleIdentifierKey       @"WIRApplicationBundleIdentifierKey"
#define WIRApplicationNameKey                   @"WIRApplicationNameKey"
#define WIRIsApplicationProxyKey                @"WIRIsApplicationProxyKey"
#define WIRIsApplicationActiveKey               @"WIRIsApplicationActiveKey"
#define WIRHostApplicationIdentifierKey         @"WIRHostApplicationIdentifierKey"
#define WIRHostApplicationNameKey               @"WIRHostApplicationNameKey"
#define WIRConnectionIdentifierKey              @"WIRConnectionIdentifierKey"
// COMPATABILITY(iOS 9): The key string is intentionally mismatched to support old relays.
#define WIRTargetIdentifierKey                  @"WIRPageIdentifierKey"
#define WIRHasLocalDebuggerKey                  @"WIRHasLocalDebuggerKey"
#define WIRTitleKey                             @"WIRTitleKey"
#define WIRURLKey                               @"WIRURLKey"
#define WIROverrideNameKey                      @"WIROverrideNameKey"
#define WIRUserInfoKey                          @"WIRUserInfoKey"
#define WIRApplicationDictionaryKey             @"WIRApplicationDictionaryKey"
#define WIRMessageDataKey                       @"WIRMessageDataKey"
#define WIRMessageDataTypeKey                   @"WIRMessageDataTypeKey"
#define WIRApplicationGetListingMessage         @"WIRApplicationGetListingMessage"
#define WIRApplicationWakeUpDebuggablesMessage  @"WIRApplicationWakeUpDebuggablesMessage"
#define WIRIndicateMessage                      @"WIRIndicateMessage"
#define WIRIndicateEnabledKey                   @"WIRIndicateEnabledKey"
#define WIRSenderKey                            @"WIRSenderKey"
#define WIRSocketDataKey                        @"WIRSocketDataKey"
#define WIRSocketDataMessage                    @"WIRSocketDataMessage"
#define WIRSocketSetupMessage                   @"WIRSocketSetupMessage"
#define WIRWebPageCloseMessage                  @"WIRWebPageCloseMessage"
#define WIRRawDataMessage                       @"WIRRawDataMessage"
#define WIRRawDataKey                           @"WIRRawDataKey"
#define WIRListingMessage                       @"WIRListingMessage"
#define WIRListingKey                           @"WIRListingKey"
#define WIRRemoteAutomationEnabledKey           @"WIRRemoteAutomationEnabledKey"
#define WIRAutomationAvailabilityKey            @"WIRAutomationAvailabilityKey"
#define WIRDestinationKey                       @"WIRDestinationKey"
#define WIRConnectionDiedMessage                @"WIRConnectionDiedMessage"
#define WIRTypeKey                              @"WIRTypeKey"
#define WIRTypeAutomation                       @"WIRTypeAutomation"
#define WIRTypeITML                             @"WIRTypeITML"
#define WIRTypeJavaScript                       @"WIRTypeJavaScript"
#define WIRTypePage                             @"WIRTypePage"
#define WIRTypeServiceWorker                    @"WIRTypeServiceWorker"
#define WIRTypeWeb                              @"WIRTypeWeb" // COMPATIBILITY (iOS 13): "Web" was split into "Page" (WebCore::Page) and "WebPage" (WebKit::WebPageProxy).
#define WIRTypeWebPage                          @"WIRTypeWebPage"
#define WIRAutomaticallyPause                   @"WIRAutomaticallyPause"
#define WIRMessageDataTypeChunkSupportedKey     @"WIRMessageDataTypeChunkSupportedKey"
#define WIRPingMessage                          @"WIRPingMessage"
#define WIRPingSuccessMessage                   @"WIRPingSuccessMessage"

// Allowed values for WIRMessageDataTypeKey.
#define WIRMessageDataTypeFull                  @"WIRMessageDataTypeFull"
#define WIRMessageDataTypeChunk                 @"WIRMessageDataTypeChunk"
#define WIRMessageDataTypeFinalChunk            @"WIRMessageDataTypeFinalChunk"

// Allowed values for WIRAutomationAvailabilityKey.
#define WIRAutomationAvailabilityNotAvailable     @"WIRAutomationAvailabilityNotAvailable"
#define WIRAutomationAvailabilityAvailable        @"WIRAutomationAvailabilityAvailable"
#define WIRAutomationAvailabilityUnknown          @"WIRAutomationAvailabilityUnknown"

#define WIRAutomaticInspectionEnabledKey           @"WIRAutomaticInspectionEnabledKey"
#define WIRAutomaticInspectionSessionIdentifierKey @"WIRAutomaticInspectionSessionIdentifierKey"
#define WIRAutomaticInspectionConfigurationMessage @"WIRAutomaticInspectionConfigurationMessage"
#define WIRAutomaticInspectionRejectMessage        @"WIRAutomaticInspectionRejectMessage"
#define WIRAutomaticInspectionCandidateMessage     @"WIRAutomaticInspectionCandidateMessage"

#define WIRAutomationTargetIsPairedKey             @"WIRAutomationTargetIsPairedKey"
#define WIRAutomationTargetNameKey                 @"WIRAutomationTargetNameKey"
#define WIRAutomationTargetVersionKey              @"WIRAutomationTargetVersionKey"
#define WIRSessionIdentifierKey                    @"WIRSessionIdentifierKey"
#define WIRSessionCapabilitiesKey                  @"WIRSessionCapabilitiesKey"
#define WIRAutomationSessionRequestMessage         @"WIRAutomationSessionRequestMessage"

// The value for WIRSessionCapabilitiesKey is a dictionary that holds these capability key-value pairs.

#define WIRAcceptInsecureCertificatesKey               @"org.webkit.webdriver.accept-insecure-certificates"
#define WIRAllowInsecureMediaCaptureCapabilityKey      @"org.webkit.webdriver.webrtc.allow-insecure-media-capture"
#define WIRSuppressICECandidateFilteringCapabilityKey  @"org.webkit.webdriver.webrtc.suppress-ice-candidate-filtering"

// These definitions are shared with a Simulator webinspectord and
// OS X process communicating with it.

#define WIRSimulatorBuildKey                    @"WIRSimulatorBuildKey"
#define WIRSimulatorProductVersionKey           @"WIRSimulatorProductVersionKey"
#define WIRSimulatorNameKey                     @"WIRSimulatorNameKey"

// These definitions are shared between webinspectord and WebKit.

#define WIRPermissionDenied                     @"WIRPermissionDenied"
#define WIRProxyApplicationParentPIDKey         @"WIRProxyApplicationParentPID"
#define WIRProxyApplicationParentAuditDataKey   @"WIRProxyApplicationParentAuditData"
#define WIRProxyApplicationSetupMessage         @"WIRProxyApplicationSetupMessage"
#define WIRProxyApplicationSetupResponseMessage @"WIRProxyApplicationSetupResponseMessage"

#define WIRRemoteInspectorEnabledKey            CFSTR("RemoteInspectorEnabled")
#define WIRRemoteInspectorDomainName            CFSTR("com.apple.webinspectord")
