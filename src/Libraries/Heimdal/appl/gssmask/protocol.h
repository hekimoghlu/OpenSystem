/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 26, 2024.
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
/*
 * $Id$
 */

/* missing from tests:
 * - export context
 * - import context
 */

/*
 * wire encodings:
 *   int16: number, 2 bytes, in network order
 *   int32: number, 4 bytes, in network order
 *   length-encoded: [int32 length, data of length bytes]
 *   string: [int32 length, string of length + 1 bytes, includes trailing '\0' ]
 */

enum gssMaggotErrorCodes {
    GSMERR_OK		= 0,
    GSMERR_ERROR,
    GSMERR_CONTINUE_NEEDED,
    GSMERR_INVALID_TOKEN,
    GSMERR_AP_MODIFIED,
    GSMERR_TEST_ISSUE,
    GSMERR_NOT_SUPPORTED
};

/*
 * input:
 *   int32: message OP (enum gssMaggotProtocol)
 *   ...
 *
 * return:   -- on error
 *    int32: not support (GSMERR_NOT_SUPPORTED)
 *
 * return:   -- on existing message OP
 *    int32: support (GSMERR_OK) -- only sent for extensions
 *    ...
 */

#define GSSMAGGOTPROTOCOL 14

enum gssMaggotOp {
    eGetVersionInfo	= 0,
    /*
     * input:
     *   none
     * return:
     *   int32: last version handled
     */
    eGoodBye,
    /*
     * input:
     *   none
     * return:
     *   close socket
     */
    eInitContext,
    /*
     * input:
     *   int32: hContext
     *   int32: hCred
     *   int32: Flags
     *      the lowest 0x7f flags maps directly to GSS-API flags
     *      DELEGATE		0x001
     *      MUTUAL_AUTH		0x002
     *      REPLAY_DETECT	0x004
     *      SEQUENCE_DETECT	0x008
     *      CONFIDENTIALITY	0x010
     *      INTEGRITY		0x020
     *      ANONYMOUS		0x040
     *
     *      FIRST_CALL		0x080
     *
     *      NTLM		0x100
     *      SPNEGO		0x200
     *   length-encoded: targetname
     *   length-encoded: token
     * return:
     *   int32: hNewContextId
     *   int32: gssapi status val
     *   length-encoded: output token
     */
    eAcceptContext,
    /*
     * input:
     *   int32: hContext
     *   int32: Flags		-- unused ?
     *      flags are same as flags for eInitContext
     *   length-encoded: token
     * return:
     *   int32: hNewContextId
     *   int32: gssapi status val
     *   length-encoded: output token
     *   int32: delegation cred id
     */
    eToastResource,
    /*
     * input:
     *   int32: hResource
     * return:
     *   int32: gsm status val
     */
    eAcquireCreds,
    /*
     * input:
     *   string: principal name
     *   string: password
     *   int32: flags
     *      FORWARDABLE		0x001
     *      DEFAULT_CREDS	0x002
     *
     *      NTLM		0x100
     *      SPNEGO		0x200
     * return:
     *   int32: gsm status val
     *   int32: hCred
     */
    eEncrypt,
    /*
     * input:
     *   int32: hContext
     *   int32: flags
     *   int32: seqno		-- unused
     *   length-encode: plaintext
     * return:
     *   int32: gsm status val
     *   length-encode: ciphertext
     */
    eDecrypt,
    /*
     * input:
     *   int32: hContext
     *   int32: flags
     *   int32: seqno		-- unused
     *   length-encode: ciphertext
     * return:
     *   int32: gsm status val
     *   length-encode: plaintext
     */
    eSign,
    /* message same as eEncrypt */
    eVerify,
    /*
     * input:
     *   int32: hContext
     *   int32: flags
     *   int32: seqno		-- unused
     *   length-encode: message
     *   length-encode: signature
     * return:
     *   int32: gsm status val
     */
    eGetVersionAndCapabilities,
    /*
     * return:
     *   int32: protocol version
     *   int32: capability flags */
#define      ISSERVER		0x01
#define      ISKDC		0x02
#define      MS_KERBEROS	0x04
#define      LOGSERVER		0x08
#define      HAS_MONIKER	0x10
    /*   string: version string
     */
    eGetTargetName,
    /*
     * return:
     *   string: target principal name
     */
    eSetLoggingSocket,
    /*
     * input:
     *   int32: hostPort
     * return to the port on the host:
     *   int32: opcode - for example eLogSetMoniker
     */
    eChangePassword,
    /* here ended version 7 of the protocol */
    /*
     * input:
     *   string: principal name
     *   string: old password
     *   string: new password
     * return:
     *   int32: gsm status val
     */
    eSetPasswordSelf,
    /* same as eChangePassword */
    eWrap,
    /* message same as eEncrypt */
    eUnwrap,
    /* message same as eDecrypt */
    eConnectLoggingService2,
    /*
     * return1:
     *   int16: log port number
     *   int32: master log prototocol version (0)
     *
     * wait for master to connect on the master log socket
     *
     * return2:
     *   int32: gsm connection status
     *   int32: maggot log prototocol version (2)
     */
    eGetMoniker,
    /*
     * return:
     *   string: moniker (Nickname the master can refer to maggot)
     */
    eCallExtension,
    /*
     * input:
     *   string: extension name
     *   int32: message id
     * return:
     *   int32: gsm status val
     */
    eAcquirePKInitCreds,
    /*
     * input:
     *   int32: flags
     *   length-encode: certificate (pkcs12 data)
     * return:
     *   int32: hResource
     *   int32: gsm status val (GSMERR_NOT_SUPPORTED)
     */
    /* here ended version 7 of the protocol */
    eWrapExt,
    /*
     * input:
     *   int32: hContext
     *   int32: flags
     *   int32: bflags
     *   length-encode: protocol header
     *   length-encode: plaintext
     *   length-encode: protocol trailer
     * return:
     *   int32: gsm status val
     *   length-encode: ciphertext
     */
    eUnwrapExt,
    /*
     * input:
     *   int32: hContext
     *   int32: flags
     *   int32: bflags
     *   length-encode: protocol header
     *   length-encode: ciphertext
     *   length-encode: protocol trailer
     * return:
     *   int32: gsm status val
     *   length-encode: plaintext
     */
    /* here ended version 8 of the protocol */

    eLastProtocolMessage
};

/* bflags */
#define WRAP_EXP_ONLY_HEADER 1

enum gssMaggotLogOp{
  eLogInfo = 0,
	/*
	string: File
	int32: Line
	string: message
     reply:
  	int32: ackid
	*/
  eLogFailure,
	/*
	string: File
	int32: Line
	string: message
     reply:
  	int32: ackid
	*/
  eLogSetMoniker
	/*
	string: moniker
	*/
};
