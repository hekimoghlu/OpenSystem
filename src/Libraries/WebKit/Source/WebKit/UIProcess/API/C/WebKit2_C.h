/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 7, 2025.
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
#ifndef WebKit2_C_h
#define WebKit2_C_h

#include <WebKit/WKBase.h>
#include <WebKit/WKType.h>

#include <WebKit/WKArray.h>
#include <WebKit/WKBackForwardListRef.h>
#include <WebKit/WKBackForwardListItemRef.h>
#include <WebKit/WKContext.h>
#include <WebKit/WKData.h>
#include <WebKit/WKDictionary.h>
#include <WebKit/WKErrorRef.h>
#include <WebKit/WKFeature.h>
#include <WebKit/WKFormSubmissionListener.h>
#include <WebKit/WKFrame.h>
#include <WebKit/WKFramePolicyListener.h>
#include <WebKit/WKGeolocationManager.h>
#include <WebKit/WKGeolocationPermissionRequest.h>
#include <WebKit/WKGeolocationPosition.h>
#include <WebKit/WKHitTestResult.h>
#include <WebKit/WKMutableArray.h>
#include <WebKit/WKMutableDictionary.h>
#include <WebKit/WKNavigationDataRef.h>
#include <WebKit/WKNumber.h>
#include <WebKit/WKOpenPanelParametersRef.h>
#include <WebKit/WKOpenPanelResultListener.h>
#include <WebKit/WKPage.h>
#include <WebKit/WKPageConfigurationRef.h>
#include <WebKit/WKPageGroup.h>
#include <WebKit/WKPreferencesRef.h>
#include <WebKit/WKString.h>
#include <WebKit/WKURL.h>
#include <WebKit/WKURLRequest.h>
#include <WebKit/WKURLResponse.h>
#include <WebKit/WKUserContentControllerRef.h>
#include <WebKit/WKUserMediaPermissionRequest.h>
#include <WebKit/WKUserScriptRef.h>

#if defined(__OBJC__) && __OBJC__
#import <WebKit/WKView.h>
#elif !(defined(__APPLE__) && __APPLE__)
#include <WebKit/WKView.h>
#endif

#endif /* WebKit2_C_h */
