/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 15, 2025.
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

#ifdef WITH_XMSS
/* $OpenBSD: xmss_fast.h,v 1.2 2018/02/26 03:56:44 dtucker Exp $ */
/*
xmss_fast.h version 20160722
Andreas HÃ¼lsing
Joost Rijneveld
Public domain.
*/

#include "xmss_wots.h"

#ifndef XMSS_H
#define XMSS_H
typedef struct{
  unsigned int level;
  unsigned long long subtree;
  unsigned int subleaf;
} leafaddr;

typedef struct{
  wots_params wots_par;
  unsigned int n;
  unsigned int h;
  unsigned int k;
} xmss_params;

typedef struct{
  xmss_params xmss_par;
  unsigned int n;
  unsigned int h;
  unsigned int d;
  unsigned int index_len;
} xmssmt_params;

typedef struct{
  unsigned int h;
  unsigned int next_idx;
  unsigned int stackusage;
  unsigned char completed;
  unsigned char *node;
} treehash_inst;

typedef struct {
  unsigned char *stack;
  unsigned int stackoffset;
  unsigned char *stacklevels;
  unsigned char *auth;
  unsigned char *keep;
  treehash_inst *treehash;
  unsigned char *retain;
  unsigned int next_leaf;
} bds_state;

/**
 * Initialize BDS state struct
 * parameter names are the same as used in the description of the BDS traversal
 */
void xmss_set_bds_state(bds_state *state, unsigned char *stack, int stackoffset, unsigned char *stacklevels, unsigned char *auth, unsigned char *keep, treehash_inst *treehash, unsigned char *retain, int next_leaf);
/**
 * Initializes parameter set.
 * Needed, for any of the other methods.
 */
int xmss_set_params(xmss_params *params, int n, int h, int w, int k);
/**
 * Initialize xmssmt_params struct
 * parameter names are the same as in the draft
 * 
 * Especially h is the total tree height, i.e. the XMSS trees have height h/d
 */
int xmssmt_set_params(xmssmt_params *params, int n, int h, int d, int w, int k);
/**
 * Generates a XMSS key pair for a given parameter set.
 * Format sk: [(32bit) idx || SK_SEED || SK_PRF || PUB_SEED || root]
 * Format pk: [root || PUB_SEED] omitting algo oid.
 */
int xmss_keypair(unsigned char *pk, unsigned char *sk, bds_state *state, xmss_params *params);
/**
 * Signs a message.
 * Returns 
 * 1. an array containing the signature followed by the message AND
 * 2. an updated secret key!
 * 
 */
int xmss_sign(unsigned char *sk, bds_state *state, unsigned char *sig_msg, unsigned long long *sig_msg_len, const unsigned char *msg,unsigned long long msglen, const xmss_params *params);
/**
 * Verifies a given message signature pair under a given public key.
 * 
 * Note: msg and msglen are pure outputs which carry the message in case verification succeeds. The (input) message is assumed to be within sig_msg which has the form (sig||msg). 
 */
int xmss_sign_open(unsigned char *msg,unsigned long long *msglen, const unsigned char *sig_msg,unsigned long long sig_msg_len, const unsigned char *pk, const xmss_params *params);

/*
 * Generates a XMSSMT key pair for a given parameter set.
 * Format sk: [(ceil(h/8) bit) idx || SK_SEED || SK_PRF || PUB_SEED || root]
 * Format pk: [root || PUB_SEED] omitting algo oid.
 */
int xmssmt_keypair(unsigned char *pk, unsigned char *sk, bds_state *states, unsigned char *wots_sigs, xmssmt_params *params);
/**
 * Signs a message.
 * Returns 
 * 1. an array containing the signature followed by the message AND
 * 2. an updated secret key!
 * 
 */
int xmssmt_sign(unsigned char *sk, bds_state *state, unsigned char *wots_sigs, unsigned char *sig_msg, unsigned long long *sig_msg_len, const unsigned char *msg, unsigned long long msglen, const xmssmt_params *params);
/**
 * Verifies a given message signature pair under a given public key.
 */
int xmssmt_sign_open(unsigned char *msg, unsigned long long *msglen, const unsigned char *sig_msg, unsigned long long sig_msg_len, const unsigned char *pk, const xmssmt_params *params);
#endif
#endif /* WITH_XMSS */
