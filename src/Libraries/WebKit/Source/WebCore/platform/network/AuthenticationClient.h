/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 17, 2023.
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
#ifndef AuthenticationClient_h
#define AuthenticationClient_h

namespace WebCore {

class AuthenticationChallenge;
class Credential;

class AuthenticationClient {
public:
    virtual void receivedCredential(const AuthenticationChallenge&, const Credential&) = 0;
    virtual void receivedRequestToContinueWithoutCredential(const AuthenticationChallenge&) = 0;
    virtual void receivedCancellation(const AuthenticationChallenge&) = 0;
    virtual void receivedRequestToPerformDefaultHandling(const AuthenticationChallenge&) = 0;
    virtual void receivedChallengeRejection(const AuthenticationChallenge&) = 0;

    void ref() { refAuthenticationClient(); }
    void deref() { derefAuthenticationClient(); }

protected:
    virtual ~AuthenticationClient() = default;

private:
    virtual void refAuthenticationClient() = 0;
    virtual void derefAuthenticationClient() = 0;
};

}

#endif
