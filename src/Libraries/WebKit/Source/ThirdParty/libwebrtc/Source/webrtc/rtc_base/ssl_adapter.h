/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 23, 2024.
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
#ifndef RTC_BASE_SSL_ADAPTER_H_
#define RTC_BASE_SSL_ADAPTER_H_

#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "rtc_base/async_socket.h"
#include "rtc_base/ssl_certificate.h"
#include "rtc_base/ssl_identity.h"
#include "rtc_base/ssl_stream_adapter.h"
#include "rtc_base/system/rtc_export.h"

namespace rtc {

class SSLAdapter;

// Class for creating SSL adapters with shared state, e.g., a session cache,
// which allows clients to resume SSL sessions to previously-contacted hosts.
// Clients should create the factory using Create(), set up the factory as
// needed using SetMode, and then call CreateAdapter to create adapters when
// needed.
class SSLAdapterFactory {
 public:
  virtual ~SSLAdapterFactory() {}

  // Specifies whether TLS or DTLS is to be used for the SSL adapters.
  virtual void SetMode(SSLMode mode) = 0;

  // Specify a custom certificate verifier for SSL.
  virtual void SetCertVerifier(SSLCertificateVerifier* ssl_cert_verifier) = 0;

  // Set the certificate this socket will present to incoming clients.
  // Takes ownership of `identity`.
  virtual void SetIdentity(std::unique_ptr<SSLIdentity> identity) = 0;

  // Choose whether the socket acts as a server socket or client socket.
  virtual void SetRole(SSLRole role) = 0;

  // Methods that control server certificate verification, used in unit tests.
  // Do not call these methods in production code.
  virtual void SetIgnoreBadCert(bool ignore) = 0;

  // Creates a new SSL adapter, but from a shared context.
  virtual SSLAdapter* CreateAdapter(Socket* socket) = 0;

  static std::unique_ptr<SSLAdapterFactory> Create();
};

// Class that abstracts a client-to-server SSL session. It can be created
// standalone, via SSLAdapter::Create, or through a factory as described above,
// in which case it will share state with other SSLAdapters created from the
// same factory.
// After creation, call StartSSL to initiate the SSL handshake to the server.
class SSLAdapter : public AsyncSocketAdapter {
 public:
  explicit SSLAdapter(Socket* socket) : AsyncSocketAdapter(socket) {}

  // Methods that control server certificate verification, used in unit tests.
  // Do not call these methods in production code.
  // TODO(juberti): Remove the opportunistic encryption mechanism in
  // BasicPacketSocketFactory that uses this function.
  virtual void SetIgnoreBadCert(bool ignore) = 0;

  virtual void SetAlpnProtocols(const std::vector<std::string>& protos) = 0;
  virtual void SetEllipticCurves(const std::vector<std::string>& curves) = 0;

  [[deprecated("Only TLS is supported by the adapter")]] virtual void SetMode(
      SSLMode mode) = 0;
  // Specify a custom certificate verifier for SSL.
  virtual void SetCertVerifier(SSLCertificateVerifier* ssl_cert_verifier) = 0;

  // Set the certificate this socket will present to incoming clients.
  // Takes ownership of `identity`.
  virtual void SetIdentity(std::unique_ptr<SSLIdentity> identity) = 0;

  // Choose whether the socket acts as a server socket or client socket.
  virtual void SetRole(SSLRole role) = 0;

  // StartSSL returns 0 if successful.
  // If StartSSL is called while the socket is closed or connecting, the SSL
  // negotiation will begin as soon as the socket connects.
  virtual int StartSSL(absl::string_view hostname) = 0;

  // When an SSLAdapterFactory is used, an SSLAdapter may be used to resume
  // a previous SSL session, which results in an abbreviated handshake.
  // This method, if called after SSL has been established for this adapter,
  // indicates whether the current session is a resumption of a previous
  // session.
  virtual bool IsResumedSession() = 0;

  // Create the default SSL adapter for this platform. On failure, returns null
  // and deletes `socket`. Otherwise, the returned SSLAdapter takes ownership
  // of `socket`.
  static SSLAdapter* Create(Socket* socket);

 private:
  // Not supported.
  int Listen(int backlog) override { RTC_CHECK(false); }
  Socket* Accept(SocketAddress* paddr) override { RTC_CHECK(false); }
};

///////////////////////////////////////////////////////////////////////////////

// Call this on the main thread, before using SSL.
// Call CleanupSSL when finished with SSL.
RTC_EXPORT bool InitializeSSL();

// Call to cleanup additional threads, and also the main thread.
RTC_EXPORT bool CleanupSSL();

}  // namespace rtc

#endif  // RTC_BASE_SSL_ADAPTER_H_
