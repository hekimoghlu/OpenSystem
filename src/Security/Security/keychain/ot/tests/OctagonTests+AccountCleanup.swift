/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 7, 2023.
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
import Foundation

#if OCTAGON

class OctagonAccountCleanupTests: OctagonTestsBase {

    @discardableResult
    func createAndSetRecoveryKey(context: OTCuttlefishContext) throws -> String {
        let cliqueConfiguration = self.createOTConfigurationContextForTests(contextID: context.contextID,
                                                                            otControl: self.otcliqueContext.otControl,
                                                                            altDSID: try XCTUnwrap(context.activeAccount?.altDSID))

        let recoveryKey = try XCTUnwrap(SecRKCreateRecoveryKeyString(nil), "should be able to create a recovery key")

        let setRecoveryKeyExpectation = self.expectation(description: "setRecoveryKeyExpectation callback occurs")
        TestsObjectiveC.setNewRecoveryKeyWithData(cliqueConfiguration, recoveryKey: recoveryKey) { _, error in
            XCTAssertNil(error, "error should be nil")
            setRecoveryKeyExpectation.fulfill()
        }
        self.wait(for: [setRecoveryKeyExpectation], timeout: 10)

        return recoveryKey
    }

    func getTwoPeersInCircle(contextID: String) throws -> OTClique {
        let initiatorContextID = contextID
        let bottlerContext = self.makeInitiatorContext(contextID: initiatorContextID)

        bottlerContext.startOctagonStateMachine()
        let fakeAccount = FakeCKAccountInfo()
        fakeAccount.accountStatus = .available
        fakeAccount.hasValidCredentials = true
        fakeAccount.accountPartition = .production
        let ckacctinfo = unsafeBitCast(fakeAccount, to: CKAccountInfo.self)

        bottlerContext.cloudkitAccountStateChange(nil, to: ckacctinfo)
        XCTAssertNoThrow(try bottlerContext.setCDPEnabled())
        self.assertEnters(context: bottlerContext, state: OctagonStateUntrusted, within: 10 * NSEC_PER_SEC)

        let clique: OTClique
        let bottlerotcliqueContext = self.createOTConfigurationContextForTests(contextID: initiatorContextID,
                                                                               otControl: self.otControl,
                                                                               altDSID: try XCTUnwrap(self.mockAuthKit.primaryAltDSID()))
        do {
            clique = try OTClique.newFriends(withContextData: bottlerotcliqueContext, resetReason: .testGenerated)
            XCTAssertNotNil(clique, "Clique should not be nil")
            XCTAssertNotNil(clique.cliqueMemberIdentifier, "Should have a member identifier after a clique newFriends call")
        } catch {
            XCTFail("Shouldn't have errored making new friends: \(error)")
            throw error
        }

        self.assertEnters(context: bottlerContext, state: OctagonStateReady, within: 10 * NSEC_PER_SEC)
        self.assertConsidersSelfTrusted(context: bottlerContext)

        let entropy = try self.loadSecret(label: clique.cliqueMemberIdentifier!)
        XCTAssertNotNil(entropy, "entropy should not be nil")

        // Fake that this peer also created some TLKShares for itself
        self.putFakeKeyHierarchiesInCloudKit()
        try self.putSelfTLKSharesInCloudKit(context: bottlerContext)

        let bottle = self.fakeCuttlefishServer.state.bottles[0]

        self.cuttlefishContext.startOctagonStateMachine()
        self.startCKAccountStatusMock()
        self.assertEnters(context: self.cuttlefishContext, state: OctagonStateUntrusted, within: 10 * NSEC_PER_SEC)

        let joinedViaBottleNotificationExpectation = XCTNSNotificationExpectation(name: NSNotification.Name(rawValue: OTJoinedViaBottle))

        // Before you call joinWithBottle, you need to call fetchViableBottles.
        let fetchViableExpectation = self.expectation(description: "fetchViableBottles callback occurs")
        self.cuttlefishContext.rpcFetchAllViableBottles(from: .default) { viable, _, error in
            XCTAssertNil(error, "should be no error fetching viable bottles")
            XCTAssert(viable?.contains(bottle.bottleID) ?? false, "The bottle we're about to restore should be viable")
            fetchViableExpectation.fulfill()
        }
        self.wait(for: [fetchViableExpectation], timeout: 10)

        let joinWithBottleExpectation = self.expectation(description: "joinWithBottle callback occurs")
        self.cuttlefishContext.join(withBottle: bottle.bottleID, entropy: entropy!, bottleSalt: self.otcliqueContext.altDSID!) { error in
            XCTAssertNil(error, "error should be nil")
            joinWithBottleExpectation.fulfill()
        }

        self.wait(for: [joinWithBottleExpectation, joinedViaBottleNotificationExpectation], timeout: 10)

        let dumpCallback = self.expectation(description: "dumpCallback callback occurs")
        self.tphClient.dump(with: try XCTUnwrap(self.cuttlefishContext.activeAccount)) { dump, _ in
            XCTAssertNotNil(dump, "dump should not be nil")
            let egoSelf = dump!["self"] as? [String: AnyObject]
            XCTAssertNotNil(egoSelf, "egoSelf should not be nil")
            let dynamicInfo = egoSelf!["dynamicInfo"] as? [String: AnyObject]
            XCTAssertNotNil(dynamicInfo, "dynamicInfo should not be nil")
            let included = dynamicInfo!["included"] as? [String]
            XCTAssertNotNil(included, "included should not be nil")
            XCTAssertEqual(included!.count, 2, "should be 2 peer ids")
            dumpCallback.fulfill()
        }
        self.wait(for: [dumpCallback], timeout: 10)

        self.verifyDatabaseMocks()
        self.assertEnters(context: self.cuttlefishContext, state: OctagonStateReady, within: 10 * NSEC_PER_SEC)
        self.assertAllCKKSViews(enter: SecCKKSZoneKeyStateReady, within: 10 * NSEC_PER_SEC)
        self.assertTLKSharesInCloudKit(receiver: self.cuttlefishContext, sender: self.cuttlefishContext)

        return clique
    }

    func testPeerCleanupAfterRemoval() throws {
        let initiatorContextID = "initiator-context-id"
        let clique = try self.getTwoPeersInCircle(contextID: initiatorContextID)
        XCTAssertNotNil(clique, "clique should not be nil")

        let bottlerContext = self.makeInitiatorContext(contextID: initiatorContextID)
        XCTAssertNotNil(bottlerContext, "bottlerContext should not be nil")

        let removeExpectation = self.expectation(description: "rpcRemoveFriends callback occurs")
        self.cuttlefishContext.rpcRemoveFriends(inClique: [clique.cliqueMemberIdentifier!]) { error in
            XCTAssertNil(error, "error should be nil")
            removeExpectation.fulfill()
        }
        self.wait(for: [removeExpectation], timeout: 10)

        var dumpCallback = self.expectation(description: "dumpCallback excluded still populated callback occurs")

        self.tphClient.dump(with: try XCTUnwrap(self.cuttlefishContext.activeAccount)) { dump, _ in
            XCTAssertNotNil(dump, "dump should not be nil")
            let egoSelf = dump!["self"] as? [String: AnyObject]
            XCTAssertNotNil(egoSelf, "egoSelf should not be nil")
            let dynamicInfo = egoSelf!["dynamicInfo"] as? [String: AnyObject]
            XCTAssertNotNil(dynamicInfo, "dynamicInfo should not be nil")
            let included = dynamicInfo!["included"] as? [String]
            XCTAssertNotNil(included, "included should not be nil")
            XCTAssertEqual(included!.count, 1, "should be 1 peer id")
            let excluded = dynamicInfo!["excluded"] as? [String]
            XCTAssertNotNil(excluded, "excluded should not be nil")
            XCTAssertEqual(excluded!.count, 1, "should be 1 peer id")

            dumpCallback.fulfill()
        }
        self.wait(for: [dumpCallback], timeout: 10)

        let container = try self.tphClient.getContainer(with: try XCTUnwrap(self.cuttlefishContext.activeAccount))
        let hasPeer = try container.moc.performAndWait {
            try container.model.hasPeer(withID: clique.cliqueMemberIdentifier!)
        }
        XCTAssertTrue(hasPeer, "model should still contain the untrusted peer")

        let dropExpectation = self.expectation(description: "dropPeerIDs callback occurs")

        self.tphClient.dropPeerIDs(with: self.cuttlefishContext.activeAccount, peerIDs: [clique.cliqueMemberIdentifier!]) { error in
            XCTAssertNil(error, "error should be nil")
            dropExpectation.fulfill()
        }
        self.wait(for: [dropExpectation], timeout: 10)

        dumpCallback = self.expectation(description: "excluded should be nil callback occurs")

        self.tphClient.dump(with: try XCTUnwrap(bottlerContext.activeAccount)) { dump, _ in
            XCTAssertNotNil(dump, "dump should not be nil")
            let egoSelf = dump!["self"] as? [String: AnyObject]
            XCTAssertNotNil(egoSelf, "egoSelf should not be nil")
            let dynamicInfo = egoSelf!["dynamicInfo"] as? [String: AnyObject]
            XCTAssertNotNil(dynamicInfo, "dynamicInfo should not be nil")
            let included = dynamicInfo!["included"] as? [String]
            XCTAssertNotNil(included, "included should not be nil")
            XCTAssertEqual(included!.count, 1, "should be 1 peer id")
            let excluded = dynamicInfo!["excluded"] as? [String]
            XCTAssertNil(excluded, "excluded should be nil")

            dumpCallback.fulfill()
        }
        self.wait(for: [dumpCallback], timeout: 10)

        self.sendContainerChangeWaitForFetch(context: self.cuttlefishContext)

        dumpCallback = self.expectation(description: "other peer removes excluded callback occurs")
        self.tphClient.dump(with: try XCTUnwrap(self.cuttlefishContext.activeAccount)) { dump, _ in
            XCTAssertNotNil(dump, "dump should not be nil")
            let egoSelf = dump!["self"] as? [String: AnyObject]
            XCTAssertNotNil(egoSelf, "egoSelf should not be nil")
            let dynamicInfo = egoSelf!["dynamicInfo"] as? [String: AnyObject]
            XCTAssertNotNil(dynamicInfo, "dynamicInfo should not be nil")
            let included = dynamicInfo!["included"] as? [String]
            XCTAssertNotNil(included, "included should not be nil")
            XCTAssertEqual(included!.count, 1, "should be 1 peer id")
            let excluded = dynamicInfo!["excluded"] as? [String]
            XCTAssertNil(excluded, "excluded should be nil")

            dumpCallback.fulfill()
        }
        self.wait(for: [dumpCallback], timeout: 10)
    }

    func testRecoveryKeyCleanupAfterRemoval() throws {
        try self.skipOnRecoveryKeyNotSupported()

        let initiatorContextID = "initiator-context-id"
        let clique = try self.getTwoPeersInCircle(contextID: initiatorContextID)
        XCTAssertNotNil(clique, "clique should not be nil")

        let bottlerContext = self.makeInitiatorContext(contextID: initiatorContextID)
        XCTAssertNotNil(bottlerContext, "bottlerContext should not be nil")

        let setExpectation = self.expectation(description: "setExpectation callback occurs")
        self.fakeCuttlefishServer.setRecoveryKeyListener = { _ in
            self.fakeCuttlefishServer.setRecoveryKeyListener = nil
            setExpectation.fulfill()
            return nil
        }

        SecCKKSSetTestSkipTLKShareHealing(true)

        let recoveryKey = try self.createAndSetRecoveryKey(context: bottlerContext)
        XCTAssertNotNil(recoveryKey, "recoveryKey should not be nil")
        self.wait(for: [setExpectation], timeout: 10)

        self.sendContainerChangeWaitForFetch(context: bottlerContext)

        var dumpCallback = self.expectation(description: "dump callback occurs")

        self.tphClient.dump(with: try XCTUnwrap(bottlerContext.activeAccount)) { dump, _ in
            XCTAssertNotNil(dump, "dump should not be nil")
            let egoSelf = dump!["self"] as? [String: AnyObject]
            XCTAssertNotNil(egoSelf, "egoSelf should not be nil")

            let stableInfo = egoSelf!["stableInfo"] as? [String: AnyObject]
            XCTAssertNotNil(stableInfo, "stableInfo should not be nil")
            XCTAssertNotNil(stableInfo!["recovery_signing_public_key"], "recoverySigningPublicKey should not be nil")

            let recoverySigningData = stableInfo!["recovery_signing_public_key"]
            XCTAssertNotNil(stableInfo!["recovery_encryption_public_key"], "recoveryEncryptionPublicKey should not be nil")
            let recoveryEncryptionData = stableInfo!["recovery_encryption_public_key"]

            XCTAssertEqual(dump!["modelRecoverySigningPublicKey"] as! String, recoverySigningData as! String, "modelRecoverySigningPublicKey should not be empty")
            XCTAssertEqual(dump!["modelRecoveryEncryptionPublicKey"] as! String, recoveryEncryptionData as! String, "modelRecoveryEncryptionPublicKey should not be empty")

            let dynamicInfo = egoSelf!["dynamicInfo"] as? [String: AnyObject]
            XCTAssertNotNil(dynamicInfo, "dynamicInfo should not be nil")
            let included = dynamicInfo!["included"] as? [String]
            XCTAssertNotNil(included, "included should not be nil")
            XCTAssertEqual(included!.count, 2, "should be 2 peer ids")
            XCTAssertNil(egoSelf!["excluded"], "excluded should be nil")

            dumpCallback.fulfill()
        }
        self.wait(for: [dumpCallback], timeout: 10)

        let removalCallback = self.expectation(description: "rpcRemoveRecoveryKey callback occurs")

        bottlerContext.rpcRemoveRecoveryKey { result, error in
            XCTAssertNil(error, "error should be nil")
            XCTAssertTrue(result, "result should be true")
            removalCallback.fulfill()
        }
        self.wait(for: [removalCallback], timeout: 10)

        self.sendContainerChangeWaitForFetch(context: bottlerContext)
        self.sendContainerChangeWaitForFetch(context: self.cuttlefishContext)

        dumpCallback = self.expectation(description: "dumpCallback excluded still populated callback occurs")

        self.tphClient.dump(with: try XCTUnwrap(self.cuttlefishContext.activeAccount)) { dump, _ in
            XCTAssertNotNil(dump, "dump should not be nil")
            let egoSelf = dump!["self"] as? [String: AnyObject]
            XCTAssertNotNil(egoSelf, "egoSelf should not be nil")
            let dynamicInfo = egoSelf!["dynamicInfo"] as? [String: AnyObject]
            XCTAssertNotNil(dynamicInfo, "dynamicInfo should not be nil")
            let included = dynamicInfo!["included"] as? [String]
            XCTAssertNotNil(included, "included should not be nil")
            XCTAssertEqual(included!.count, 2, "should be 2 peer ids")
            let excluded = dynamicInfo!["excluded"] as? [String]
            XCTAssertNotNil(excluded, "excluded should not be nil")
            XCTAssertEqual(excluded!.count, 1, "should be 1 peer id")

            let recoveryKeyPeerID = excluded![0]
            XCTAssertTrue(recoveryKeyPeerID.contains("RK-"), "should contain excluded recovery key peerID")

            dumpCallback.fulfill()
        }
        self.wait(for: [dumpCallback], timeout: 10)

        self.sendContainerChangeWaitForFetch(context: bottlerContext)
        self.sendContainerChangeWaitForFetch(context: self.cuttlefishContext)

        dumpCallback = self.expectation(description: "dumpCallback excluded still populated callback occurs")

        self.tphClient.dump(with: try XCTUnwrap(bottlerContext.activeAccount)) { dump, _ in
            XCTAssertNotNil(dump, "dump should not be nil")
            let egoSelf = dump!["self"] as? [String: AnyObject]
            XCTAssertNotNil(egoSelf, "egoSelf should not be nil")
            let dynamicInfo = egoSelf!["dynamicInfo"] as? [String: AnyObject]
            XCTAssertNotNil(dynamicInfo, "dynamicInfo should not be nil")
            let included = dynamicInfo!["included"] as? [String]
            XCTAssertNotNil(included, "included should not be nil")
            XCTAssertEqual(included!.count, 2, "should be 2 peer ids")
            let excluded = dynamicInfo!["excluded"] as? [String]
            XCTAssertNotNil(excluded, "excluded should not be nil")
            XCTAssertEqual(excluded!.count, 1, "should be 1 peer id")

            let recoveryKeyPeerID = excluded![0]
            XCTAssertTrue(recoveryKeyPeerID.contains("RK-"), "should contain excluded recovery key peerID")

            dumpCallback.fulfill()
        }
        self.wait(for: [dumpCallback], timeout: 10)

        let removeExpectation = self.expectation(description: "rpcRemoveFriends callback occurs")
        self.cuttlefishContext.rpcRemoveFriends(inClique: [clique.cliqueMemberIdentifier!]) { error in
            XCTAssertNil(error, "error should be nil")
            removeExpectation.fulfill()
        }
        self.wait(for: [removeExpectation], timeout: 10)

        let healthCheckCallback = self.expectation(description: "healthCheckCallback callback occurs")
        self.manager.healthCheck(OTControlArguments(configuration: self.otcliqueContext), skipRateLimitingCheck: false, repair: false) { response, error in
            XCTAssertNil(error, "error should be nil")
            XCTAssertNotNil(response, "response should not be nil")
            healthCheckCallback.fulfill()
        }
        self.wait(for: [healthCheckCallback], timeout: 10)

        self.sendContainerChangeWaitForFetch(context: bottlerContext)
        self.sendContainerChangeWaitForFetch(context: self.cuttlefishContext)

        dumpCallback = self.expectation(description: "dumpCallback excluded still populated callback occurs")

        self.tphClient.dump(with: try XCTUnwrap(self.cuttlefishContext.activeAccount)) { dump, _ in
            XCTAssertNotNil(dump, "dump should not be nil")
            let egoSelf = dump!["self"] as? [String: AnyObject]
            XCTAssertNotNil(egoSelf, "egoSelf should not be nil")
            let dynamicInfo = egoSelf!["dynamicInfo"] as? [String: AnyObject]
            XCTAssertNotNil(dynamicInfo, "dynamicInfo should not be nil")
            let included = dynamicInfo!["included"] as? [String]
            XCTAssertNotNil(included, "included should not be nil")
            XCTAssertEqual(included!.count, 1, "should be 1 peer id")
            let excluded = dynamicInfo!["excluded"] as? [String]
            XCTAssertNotNil(excluded, "excluded should not be nil")
            XCTAssertEqual(excluded!.count, 2, "should be 2 peer id")

            XCTAssertTrue(excluded![0].contains("RK-") || excluded![1].contains("RK-"), "should contain excluded recovery key peerID")

            dumpCallback.fulfill()
        }
        self.wait(for: [dumpCallback], timeout: 10)

        let dropExpectation = self.expectation(description: "dropPeerIDs callback occurs")

        self.tphClient.dropPeerIDs(with: self.cuttlefishContext.activeAccount, peerIDs: [clique.cliqueMemberIdentifier!]) { error in
            XCTAssertNil(error, "error should be nil")
            dropExpectation.fulfill()
        }
        self.wait(for: [dropExpectation], timeout: 10)

        dumpCallback = self.expectation(description: "dumpCallback excluded still populated callback occurs")

        self.tphClient.dump(with: try XCTUnwrap(self.cuttlefishContext.activeAccount)) { dump, _ in
            XCTAssertNotNil(dump, "dump should not be nil")
            let egoSelf = dump!["self"] as? [String: AnyObject]
            XCTAssertNotNil(egoSelf, "egoSelf should not be nil")
            let dynamicInfo = egoSelf!["dynamicInfo"] as? [String: AnyObject]
            XCTAssertNotNil(dynamicInfo, "dynamicInfo should not be nil")
            let included = dynamicInfo!["included"] as? [String]
            XCTAssertNotNil(included, "included should not be nil")
            XCTAssertEqual(included!.count, 1, "should be 1 peer id")
            let excluded = dynamicInfo!["excluded"] as? [String]
            XCTAssertNotNil(excluded, "excluded should not be nil")
            XCTAssertEqual(excluded!.count, 2, "should be 2 peer id")

            XCTAssertTrue(excluded![0].contains("RK-") || excluded![1].contains("RK-"), "should contain excluded recovery key peerID")

            dumpCallback.fulfill()
        }
        self.wait(for: [dumpCallback], timeout: 10)

        self.sendContainerChangeWaitForUntrustedFetch(context: bottlerContext)
        self.sendContainerChangeWaitForFetch(context: self.cuttlefishContext)

        dumpCallback = self.expectation(description: "dumpCallback excluded still populated callback occurs")

        self.tphClient.dump(with: try XCTUnwrap(bottlerContext.activeAccount)) { dump, _ in
            XCTAssertNotNil(dump, "dump should not be nil")
            let egoSelf = dump!["self"] as? [String: AnyObject]
            XCTAssertNotNil(egoSelf, "egoSelf should not be nil")
            let dynamicInfo = egoSelf!["dynamicInfo"] as? [String: AnyObject]
            XCTAssertNotNil(dynamicInfo, "dynamicInfo should not be nil")
            let included = dynamicInfo!["included"] as? [String]
            XCTAssertNil(included, "included should be nil")
            let excluded = dynamicInfo!["excluded"] as? [String]
            XCTAssertNotNil(excluded, "excluded should not be nil")
            XCTAssertEqual(excluded!.count, 1, "should be 1 peer id")
            XCTAssertEqual(egoSelf!["peerID"] as! String, excluded![0], "excluded peer should be the self peer")
            dumpCallback.fulfill()
        }
        self.wait(for: [dumpCallback], timeout: 10)
    }
}

#endif
