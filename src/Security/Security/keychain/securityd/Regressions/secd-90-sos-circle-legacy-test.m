/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 23, 2022.
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
#include <Security/SecBase.h>
#include <Security/SecItem.h>
#include <Security/SecKey.h>
#include <Security/SecKeyPriv.h>
#include "keychain/SecureObjectSync/SOSCircle.h"
#include "keychain/SecureObjectSync/SOSCirclePriv.h"
#include <Security/SecureObjectSync/SOSPeerInfo.h>
#include "keychain/SecureObjectSync/SOSPeerInfoInternal.h"
#include "keychain/SecureObjectSync/SOSPeerInfoCollections.h"
#include "keychain/SecureObjectSync/SOSInternal.h"
#include "keychain/SecureObjectSync/SOSUserKeygen.h"
#include <utilities/SecCFWrappers.h>
#include <CoreFoundation/CoreFoundation.h>
#include <stdlib.h>
#include <unistd.h>
#include "SOSAccountTesting.h"
#include "SOSRegressionUtilities.h"
#include "SecdTestKeychainUtilities.h"
#include "secd_regressions.h"

#if SOS_ENABLED


static void tests(void)
{
    SecKeyRef correctUserKey = SOSMakeUserKeyForPassword("theCorrectPassword");
    ok(correctUserKey, "made a userKey");
    SecKeyRef badUserKey = SOSMakeUserKeyForPassword("theBadPassword");
    ok(correctUserKey, "made a userKey to use for an invalid one");

    // Test PeerInfo sensing
    SOSFullPeerInfoRef newMacOSPeer = SOSTestFullPeerInfo(CFSTR("newMacOSPeer"), correctUserKey, CFSTR("21A132"), SOSPeerInfo_macOS);
    ok(!SOSPeerInfoIsLegacy(SOSFullPeerInfoGetPeerInfo(newMacOSPeer)), "this peer should be modern");
    ok(SOSPeerValidityCheck(newMacOSPeer, correctUserKey, NULL), "this should verify");

    // SOSPeerInfoIsLegacy only checks OS and devtype.  A peer signed with an old/different userKey is still evaluated on those criteria.
    // Validity is needed to pass the circle test
    SOSFullPeerInfoRef newInvalidMacOSPeer = SOSTestFullPeerInfo(CFSTR("newInvalidMacOSPeer"), badUserKey, CFSTR("21A132"), SOSPeerInfo_macOS);
    ok(!SOSPeerInfoIsLegacy(SOSFullPeerInfoGetPeerInfo(newInvalidMacOSPeer)), "this peer should be modern");
    ok(!SOSPeerValidityCheck(newInvalidMacOSPeer, correctUserKey, NULL), "this shouldn't verify");
    
    SOSFullPeerInfoRef oldMacOSPeer = SOSTestFullPeerInfo(CFSTR("oldMacOSPeer"), correctUserKey, CFSTR("20A13"), SOSPeerInfo_macOS);
    ok(SOSPeerInfoIsLegacy(SOSFullPeerInfoGetPeerInfo(oldMacOSPeer)), "this peer should be legacy");
    ok(SOSPeerValidityCheck(oldMacOSPeer, correctUserKey, NULL), "this should verify");

    SOSFullPeerInfoRef oldInvalidMacOSPeer = SOSTestFullPeerInfo(CFSTR("oldInvalidMacOSPeer"), badUserKey, CFSTR("20A13"), SOSPeerInfo_macOS);
    ok(SOSPeerInfoIsLegacy(SOSFullPeerInfoGetPeerInfo(oldInvalidMacOSPeer)), "this peer should be legacy");
    ok(!SOSPeerValidityCheck(oldInvalidMacOSPeer, correctUserKey, NULL), "this shouldn't verify");

    SOSFullPeerInfoRef newIOSPeer = SOSTestFullPeerInfo(CFSTR("newIOSPeer"), correctUserKey, CFSTR("19B2347"), SOSPeerInfo_iOS);
    ok(!SOSPeerInfoIsLegacy(SOSFullPeerInfoGetPeerInfo(newIOSPeer)), "this peer should be modern");
    ok(SOSPeerValidityCheck(newIOSPeer, correctUserKey, NULL), "this should verify");

    // SOSPeerInfoIsLegacy only checks OS and devtype.  A peer signed with an old/different userKey is still evaluated on those criteria.
    // Validity is needed to pass the circle test
    SOSFullPeerInfoRef newInvalidIOSPeer = SOSTestFullPeerInfo(CFSTR("newInvalidIOSPeer"), badUserKey, CFSTR("19B2347"), SOSPeerInfo_iOS);
    ok(!SOSPeerInfoIsLegacy(SOSFullPeerInfoGetPeerInfo(newInvalidIOSPeer)), "this peer should be modern");
    ok(!SOSPeerValidityCheck(newInvalidIOSPeer, correctUserKey, NULL), "this shouldn't verify");

    SOSFullPeerInfoRef oldIOSPeer = SOSTestFullPeerInfo(CFSTR("oldIOSPeer"), correctUserKey, CFSTR("17C47"), SOSPeerInfo_iOS);
    ok(SOSPeerInfoIsLegacy(SOSFullPeerInfoGetPeerInfo(oldIOSPeer)), "this peer should be legacy");
    ok(SOSPeerValidityCheck(oldIOSPeer, correctUserKey, NULL), "this should verify");

    SOSFullPeerInfoRef oldInvalidIOSPeer = SOSTestFullPeerInfo(CFSTR("oldInvalidMacOSPeer"), badUserKey, CFSTR("17C47"), SOSPeerInfo_iOS);
    ok(SOSPeerInfoIsLegacy(SOSFullPeerInfoGetPeerInfo(oldInvalidIOSPeer)), "this peer should be legacy");
    ok(!SOSPeerValidityCheck(oldInvalidIOSPeer, correctUserKey, NULL), "this shouldn't verify");

    // iCloud Identities can be old and still not legacy
    SOSFullPeerInfoRef iCloudIdentityPeer = SOSTestFullPeerInfo(CFSTR("iCloudIdentityPeer"), correctUserKey, CFSTR("17C47"), SOSPeerInfo_iCloud);
    ok(!SOSPeerInfoIsLegacy(SOSFullPeerInfoGetPeerInfo(iCloudIdentityPeer)), "this peer should be modern");
    ok(SOSPeerValidityCheck(iCloudIdentityPeer, correctUserKey, NULL), "this should verify");

    // really old peers are unknown
    SOSFullPeerInfoRef unknownPeer = SOSTestFullPeerInfo(CFSTR("unknownPeer"), correctUserKey, CFSTR("UNKNOWN"), SOSPeerInfo_unknown);
    ok(SOSPeerInfoIsLegacy(SOSFullPeerInfoGetPeerInfo(unknownPeer)), "this peer should be legacy");
    ok(SOSPeerValidityCheck(unknownPeer, correctUserKey, NULL), "this should verify");

    // we've never had watches before for SOS - all flavors are modern
    SOSFullPeerInfoRef watchPeer = SOSTestFullPeerInfo(CFSTR("watchPeer"), correctUserKey, CFSTR("17T47"), SOSPeerInfo_watchOS);
    ok(!SOSPeerInfoIsLegacy(SOSFullPeerInfoGetPeerInfo(watchPeer)), "this peer should be modern");
    ok(SOSPeerValidityCheck(watchPeer, correctUserKey, NULL), "this should verify");

    // we've never had appletvs before for SOS - all flavors are modern
    SOSFullPeerInfoRef appletvPeer = SOSTestFullPeerInfo(CFSTR("appletvPeer"), correctUserKey, CFSTR("18K785"), SOSPeerInfo_tvOS);
    ok(!SOSPeerInfoIsLegacy(SOSFullPeerInfoGetPeerInfo(appletvPeer)), "this peer should be modern");
    ok(SOSPeerValidityCheck(appletvPeer, correctUserKey, NULL), "this should verify");

    // Test Circle Sensing
    SOSCircleRef singleCircle = SOSTestCircle(correctUserKey, newIOSPeer, NULL);
    ok(!SOSCircleIsLegacy(singleCircle, correctUserKey), "this circle should be modern");
    CFReleaseNull(singleCircle);

    SOSCircleRef legacyCircle = SOSTestCircle(correctUserKey, oldMacOSPeer, oldIOSPeer, iCloudIdentityPeer, NULL);
    ok(SOSCircleIsLegacy(legacyCircle, correctUserKey), "this circle should be legacy");
    CFReleaseNull(legacyCircle);

    SOSCircleRef nonLegacyCircle = SOSTestCircle(correctUserKey, newMacOSPeer, newIOSPeer, iCloudIdentityPeer, NULL);
    ok(!SOSCircleIsLegacy(nonLegacyCircle, correctUserKey), "this circle should be modern");
    CFReleaseNull(nonLegacyCircle);
    
    SOSCircleRef mixedLegacyCircle = SOSTestCircle(correctUserKey, newMacOSPeer, oldIOSPeer, iCloudIdentityPeer, NULL);
    ok(SOSCircleIsLegacy(mixedLegacyCircle, correctUserKey), "this circle should be legacy");
    CFReleaseNull(mixedLegacyCircle);
    
    SOSCircleRef futuristicCircle = SOSTestCircle(correctUserKey, newMacOSPeer, newIOSPeer, watchPeer, appletvPeer, iCloudIdentityPeer, NULL);
    ok(!SOSCircleIsLegacy(futuristicCircle, correctUserKey), "this circle should be modern");
    CFReleaseNull(futuristicCircle);
    
    SOSCircleRef mixedWithInvalidOldies = SOSTestCircle(correctUserKey, newMacOSPeer, newIOSPeer, newInvalidMacOSPeer, oldInvalidMacOSPeer, iCloudIdentityPeer, NULL);
    ok(!SOSCircleIsLegacy(mixedWithInvalidOldies, correctUserKey), "this circle should be modern");
    CFReleaseNull(mixedWithInvalidOldies);
    CFReleaseNull(newMacOSPeer);
    CFReleaseNull(newInvalidMacOSPeer);
    CFReleaseNull(oldMacOSPeer);
    CFReleaseNull(oldInvalidMacOSPeer);
    CFReleaseNull(newIOSPeer);
    CFReleaseNull(newInvalidIOSPeer);
    CFReleaseNull(oldIOSPeer);
    CFReleaseNull(iCloudIdentityPeer);
    CFReleaseNull(unknownPeer);
    CFReleaseNull(watchPeer);
    CFReleaseNull(appletvPeer);
}

#endif

int secd_90_sos_circle_legacy_test(int argc, char *const *argv) {
#if SOS_ENABLED
    plan_tests(33);
    enableSOSCompatibilityForTests();
    secd_test_setup_temp_keychain(__FUNCTION__, NULL); // if the test has been run on this system this will generate one more "ok" test
    tests();
    secd_test_teardown_delete_temp_keychain(__FUNCTION__);
#else
    plan_tests(0);
#endif
    return 0;
}
