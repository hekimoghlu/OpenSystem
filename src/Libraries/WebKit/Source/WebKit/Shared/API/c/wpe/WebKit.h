/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 19, 2023.
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
#ifndef WPE_WebKit_h
#define WPE_WebKit_h

#define BUILDING_WPE__

#include <WebKit/WKBase.h>
#include <WebKit/WKType.h>

// From Source/WebKit/Shared/API/c/
#include <WebKit/WKArray.h>
#include <WebKit/WKData.h>
#include <WebKit/WKDeclarationSpecifiers.h>
#include <WebKit/WKDictionary.h>
#include <WebKit/WKErrorRef.h>
#include <WebKit/WKGeometry.h>
#include <WebKit/WKMutableArray.h>
#include <WebKit/WKMutableDictionary.h>
#include <WebKit/WKNumber.h>
#include <WebKit/WKSecurityOriginRef.h>
#include <WebKit/WKString.h>
#include <WebKit/WKURL.h>
#include <WebKit/WKURLRequest.h>
#include <WebKit/WKURLResponse.h>
#include <WebKit/WKUserContentInjectedFrames.h>
#include <WebKit/WKUserScriptInjectionTime.h>

// From Source/WebKit/WebProcess/InjectedBundle/API/c/
#include <WebKit/WKBundle.h>
#include <WebKit/WKBundleBackForwardList.h>
#include <WebKit/WKBundleBackForwardListItem.h>
#include <WebKit/WKBundleFileHandleRef.h>
#include <WebKit/WKBundleFrame.h>
#include <WebKit/WKBundleHitTestResult.h>
#include <WebKit/WKBundleInitialize.h>
#include <WebKit/WKBundleNavigationAction.h>
#include <WebKit/WKBundleNodeHandle.h>
#include <WebKit/WKBundlePage.h>
#include <WebKit/WKBundlePageBanner.h>
#include <WebKit/WKBundlePageContextMenuClient.h>
#include <WebKit/WKBundlePageEditorClient.h>
#include <WebKit/WKBundlePageFormClient.h>
#include <WebKit/WKBundlePageLoaderClient.h>
#include <WebKit/WKBundlePageOverlay.h>
#include <WebKit/WKBundlePagePolicyClient.h>
#include <WebKit/WKBundlePageResourceLoadClient.h>
#include <WebKit/WKBundlePageUIClient.h>
#include <WebKit/WKBundleRangeHandle.h>
#include <WebKit/WKBundleScriptWorld.h>

// From Source/WebKit/UIProcess/API/C
#include <WebKit/WKBackForwardListItemRef.h>
#include <WebKit/WKBackForwardListRef.h>
#include <WebKit/WKContext.h>
#include <WebKit/WKContextConfigurationRef.h>
#include <WebKit/WKCredential.h>
#include <WebKit/WKCredentialTypes.h>
#include <WebKit/WKFrame.h>
#include <WebKit/WKFrameInfoRef.h>
#include <WebKit/WKFramePolicyListener.h>
#include <WebKit/WKHitTestResult.h>
#include <WebKit/WKNavigationActionRef.h>
#include <WebKit/WKNavigationDataRef.h>
#include <WebKit/WKNavigationRef.h>
#include <WebKit/WKNavigationResponseRef.h>
#include <WebKit/WKPage.h>
#include <WebKit/WKPageConfigurationRef.h>
#include <WebKit/WKPageGroup.h>
#include <WebKit/WKPreferencesRef.h>
#include <WebKit/WKSessionStateRef.h>
#include <WebKit/WKUserContentControllerRef.h>
#include <WebKit/WKUserScriptRef.h>
#include <WebKit/WKView.h>
#include <WebKit/WKViewportAttributes.h>
#include <WebKit/WKWindowFeaturesRef.h>

#endif // WPE_WebKit_h
