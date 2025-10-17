/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 30, 2024.
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

/*
 Generate magic symbols which inform dyld of the previous install path of a symbol, for binary
 compatibility with clients which were built before Webkit's swift overlay was merged into the
 main framework.

 Uses platform constants from <mach-o/loader.h> without including it, since the constants conflict
 with <wtf/Platform.h> macros.
 */

#include "WKDeclarationSpecifiers.h"

#define MIGRATE_SYMBOL(InstallName, Platform, Introduced, Migrated, Symbol) \
    extern WK_EXPORT const char migrated_symbol_##Symbol __asm("$ld$previous$" InstallName "$$" #Platform "$" #Introduced "$" #Migrated "$" #Symbol "$") \

#if PLATFORM(MAC)
#define DECLARE_MIGRATED_NAME(Symbol, macOSVersion, iOSVersion, visionOSVersion) \
    MIGRATE_SYMBOL("/usr/lib/swift/libswiftWebKit.dylib", 1 /*PLATFORM_MACOS*/, macOSVersion, 15.4, Symbol)

#elif PLATFORM(MACCATALYST)
#define DECLARE_MIGRATED_NAME(Symbol, macOSVersion, iOSVersion, visionOSVersion) \
    MIGRATE_SYMBOL("/System/iOSSupport/usr/lib/swift/libswiftWebKit.dylib", 6 /*PLATFORM_MACCATALYST*/, iOSVersion, 18.4, Symbol)

#elif PLATFORM(IOS) && !PLATFORM(IOS_SIMULATOR)
#define DECLARE_MIGRATED_NAME(Symbol, macOSVersion, iOSVersion, visionOSVersion) \
    MIGRATE_SYMBOL("/usr/lib/swift/libswiftWebKit.dylib", 2 /*PLATFORM_IOS*/, iOSVersion, 18.4, Symbol)

#elif PLATFORM(IOS) && PLATFORM(IOS_SIMULATOR)
#define DECLARE_MIGRATED_NAME(Symbol, macOSVersion, iOSVersion, visionOSVersion) \
    MIGRATE_SYMBOL("/usr/lib/swift/libswiftWebKit.dylib", 7 /*PLATFORM_IOSSIMULATOR*/, iOSVersion, 18.4, Symbol)

#elif PLATFORM(VISION) && !PLATFORM(IOS_FAMILY_SIMULATOR)
#define DECLARE_MIGRATED_NAME(Symbol, macOSVersion, iOSVersion, visionOSVersion) \
    MIGRATE_SYMBOL("/usr/lib/swift/libswiftWebKit.dylib", 11 /*PLATFORM_VISIONOS*/, visionOSVersion, 2.4, Symbol)

#elif PLATFORM(VISION) && PLATFORM(IOS_FAMILY_SIMULATOR)
#define DECLARE_MIGRATED_NAME(Symbol, macOSVersion, iOSVersion, visionOSVersion) \
    MIGRATE_SYMBOL("/usr/lib/swift/libswiftWebKit.dylib", 12 /*PLATFORM_VISIONOSSIMULATOR*/, visionOSVersion, 2.4, Symbol)

#endif

#define FOR_EACH_MIGRATED_SWIFT_OVERLAY_SYMBOL(X) \
    X(_$sSo18WKPDFConfigurationC6WebKitE4rectSo6CGRectVSgvM, 10.16, 14.0, 1.0); \
    X(_$sSo18WKPDFConfigurationC6WebKitE4rectSo6CGRectVSgvg, 10.16, 14.0, 1.0); \
    X(_$sSo18WKPDFConfigurationC6WebKitE4rectSo6CGRectVSgvpMV, 10.16, 14.0, 1.0); \
    X(_$sSo18WKPDFConfigurationC6WebKitE4rectSo6CGRectVSgvs, 10.16, 14.0, 1.0); \
    X(_$sSo18WKWebsiteDataStoreC6WebKitE19proxyConfigurationsSay7Network18ProxyConfigurationVGvM, 10.16, 14.0, 1.0); \
    X(_$sSo18WKWebsiteDataStoreC6WebKitE19proxyConfigurationsSay7Network18ProxyConfigurationVGvg, 10.16, 14.0, 1.0); \
    X(_$sSo18WKWebsiteDataStoreC6WebKitE19proxyConfigurationsSay7Network18ProxyConfigurationVGvpMV, 10.16, 14.0, 1.0); \
    X(_$sSo18WKWebsiteDataStoreC6WebKitE19proxyConfigurationsSay7Network18ProxyConfigurationVGvs, 10.16, 14.0, 1.0); \
    X(_$sSo9WKWebViewC6WebKitE06createC11ArchiveData17completionHandleryys6ResultOy10Foundation0G0Vs5Error_pGc_tF, 10.16, 14.0, 1.0); \
    X(_$sSo9WKWebViewC6WebKitE18evaluateJavaScript_2in12contentWorldypSgSS_So11WKFrameInfoCSgSo09WKContentJ0CtYaKF, 10.16, 14.0, 1.0); \
    X(_$sSo9WKWebViewC6WebKitE18evaluateJavaScript_2in12contentWorldypSgSS_So11WKFrameInfoCSgSo09WKContentJ0CtYaKFTu, 10.16, 14.0, 1.0); \
    X(_$sSo9WKWebViewC6WebKitE18evaluateJavaScript_2inAE17completionHandlerySS_So11WKFrameInfoCSgSo14WKContentWorldCys6ResultOyyps5Error_pGcSgtF, 10.16, 14.0, 1.0); \
    X(_$sSo9WKWebViewC6WebKitE19callAsyncJavaScript_9arguments2in12contentWorldypSgSS_SDySSypGSo11WKFrameInfoCSgSo09WKContentL0CtYaKF, 10.16, 14.0, 1.0); \
    X(_$sSo9WKWebViewC6WebKitE19callAsyncJavaScript_9arguments2in12contentWorldypSgSS_SDySSypGSo11WKFrameInfoCSgSo09WKContentL0CtYaKFTu, 10.16, 14.0, 1.0); \
    X(_$sSo9WKWebViewC6WebKitE19callAsyncJavaScript_9arguments2inAF17completionHandlerySS_SDySSypGSo11WKFrameInfoCSgSo14WKContentWorldCys6ResultOyyps5Error_pGcSgtF, 10.16, 14.0, 1.0); \
    X(_$sSo9WKWebViewC6WebKitE3pdf13configuration10Foundation4DataVSo18WKPDFConfigurationC_tYaKF, 10.16, 14.0, 1.0); \
    X(_$sSo9WKWebViewC6WebKitE3pdf13configuration10Foundation4DataVSo18WKPDFConfigurationC_tYaKFTu, 10.16, 14.0, 1.0); \
    X(_$sSo9WKWebViewC6WebKitE4find_13configuration17completionHandlerySS_So19WKFindConfigurationCySo0I6ResultCctF, 10.16, 14.0, 1.0); \
    X(_$sSo9WKWebViewC6WebKitE4find_13configurationSo12WKFindResultCSS_So0G13ConfigurationCtYaKF, 10.16, 14.0, 1.0); \
    X(_$sSo9WKWebViewC6WebKitE4find_13configurationSo12WKFindResultCSS_So0G13ConfigurationCtYaKFTu, 10.16, 14.0, 1.0); \
    X(_$sSo9WKWebViewC6WebKitE9createPDF13configuration17completionHandlerySo18WKPDFConfigurationC_ys6ResultOy10Foundation4DataVs5Error_pGctF, 10.16, 14.0, 1.0); \
    X(_$sSo21WKWebExtensionContextC6WebKitE10didMoveTab_4from2inySo0abH0_p_SiSo0aB6Window_pSgtF, 15.4, 18.4, 2.4); \
    X(_$sSo21WKWebExtensionContextC6WebKitE11didCloseTab_15windowIsClosingySo0abH0_p_SbtF, 15.4, 18.4, 2.4); \
    X(_$sSo21WKWebExtensionContextC6WebKitE11didCloseTab_15windowIsClosingySo0abH0_p_SbtFfA0_, 15.4, 18.4, 2.4); \
    X(_$sSo21WKWebExtensionContextC6WebKitE14didActivateTab_014previousActiveH0ySo0babH0_p_SoAF_pSgtF, 15.4, 18.4, 2.4); \
    X(_$sSo24WKWebExtensionControllerC6WebKitE10didMoveTab_4from2inySo0abH0_p_SiSo0aB6Window_pSgtF, 15.4, 18.4, 2.4); \
    X(_$sSo24WKWebExtensionControllerC6WebKitE11didCloseTab_15windowIsClosingySo0abH0_p_SbtF, 15.4, 18.4, 2.4); \
    X(_$sSo24WKWebExtensionControllerC6WebKitE11didCloseTab_15windowIsClosingySo0abH0_p_SbtFfA0_, 15.4, 18.4, 2.4); \
    X(_$sSo24WKWebExtensionControllerC6WebKitE14didActivateTab_014previousActiveH0ySo0abH0_p_SoAF_pSgtF, 15.4, 18.4, 2.4); \
    X(__swift_FORCE_LOAD_$_swiftWebKit, 10.16, 14.0, 1.0) \


#if defined(DECLARE_MIGRATED_NAME)
FOR_EACH_MIGRATED_SWIFT_OVERLAY_SYMBOL(DECLARE_MIGRATED_NAME);

// Swift may have generated this symbol in clients of libswiftWebKit, to force the library to be
// loaded at runtime.
WK_EXPORT char _swift_FORCE_LOAD_$_swiftWebKit;
#undef DECLARE_MIGRATED_NAME
#endif

#undef MIGRATE_SYMBOL
