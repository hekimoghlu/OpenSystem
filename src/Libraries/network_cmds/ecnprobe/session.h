/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 1, 2023.
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
#define MAXREQUESTLEN 1000

#define SESSION_DEBUG_LOW	1
#define	SESSION_DEBUG_MEDIUM	2
#define	SESSION_DEBUG_HIGH	3

struct TcpSession {

  /* target name, as specified by the user */
  char targetName[MAXHOSTNAMELEN];
  
  /* DNS name of hosts */
  char targetHostName[MAXHOSTNAMELEN];	
  char sourceHostName[MAXHOSTNAMELEN];

  /* raw socket we use to send on */
  int socket;		
  
  /* connection endpoint identifiers */
  u_int32_t src;
  u_int16_t sport;
  u_int32_t dst;
  u_int16_t dport;

  /* sender info, from RFC 793 */
  u_int32_t iss;     // initial send sequence
  u_int32_t snd_una; // sequence numbers of unacknowledged data
  u_int32_t snd_nxt; // sequence number to be sent next
  u_int16_t snd_wnd; 
  u_int16_t sndmss;

  /* Receiver info */
  u_int32_t irs;
  u_int32_t rcv_wnd;
  u_int32_t rcv_nxt;
  u_int32_t maxseqseen;
  u_int16_t mss;

  /* timing */
  double rtt;
  u_int8_t ttl;
  double start_time;

  /* data buffer */
  u_int8_t *dataRcvd ;
	
  /* basic results */
  int totSent; 
  int totRcvd;
  int totSeenSent;
  int totDataPktsRcvd; 
  int totOutofSeq; 
  int hsz; 
  
  /* basic control*/
  int epochTime; 
  int debug; 
  int verbose; 
  int initSession; 
  int initCapture; 
  int initFirewall; 
  int firewall_rule_number;
  char *filename;
  int maxpkts; 

  /* New loss-rate parameters */
  float loss_rate;
  float prop_delay;

  /* results are suspect for various reasons */
  int rtt_unreliable;
  int ignore_result;

  /* Drops and reordering startistics */
  int num_reordered;
  int num_unwanted_drops;
  int num_rtos;
  int num_reord_ret;
  int num_dup_transmissions;
  int num_dup_acks;
  int num_pkts_0_dup_acks;
  int num_pkts_1_dup_acks;
  int num_pkts_2_dup_acks;
  int num_pkts_3_dup_acks;
  int num_pkts_4_or_more_dup_acks;
  int num_dupack_ret;

  /* For PMTUD test */
  int mtu;

  /* For ByteCounting test */
  int bytecounting_type;
  int ack_bytes;  /* How many bytes covered per ACK */
  int ack_rate;   /* ACK [every | every other | every third |...] packet */

  /* For WindowScale Option test */
  u_int8_t receiving_shift_count;
  u_int8_t sending_shift_count;

  /* For MidBoxTTL test */
  int curr_ttl;

  int dont_send_reset;
};

//void SendSessionPacket(struct IPPacket *packet, 
void SendSessionPacket(struct IPPacket *packet, 
		       u_int16_t ip_len, /* Total size of IP datagram */
		       u_int8_t tcp_flags,
		       u_int16_t ip_optlen, /* IP options len - New */
		       u_int16_t optlen,    /* TCP options len */
		       u_int8_t iptos);

void SendICMPReply(struct IPPacket *p);

void SendPkt(struct IPPacket *p, u_int16_t ip_len, int ip_optlen, int tcp_optlen);

void SendICMPPkt(struct ICMPUnreachableErrorPacket *p, u_int16_t ip_len);

void StorePacket (struct IPPacket *p); 

int EstablishSession(u_int32_t sourceAddress, \
		     u_int16_t sourcePort, \
		     u_int32_t targetAddress,
		     u_int16_t targetPort, \
		     int ip_optlen,\
		     char *ip_opt,\
		     int mss, 
		     int optlen, 
		     char *opt, \
		     int maxwin, 
		     int maxpkts, 
		     u_int8_t iptos, 
		     u_int8_t tcp_flags);

void rcvData (void (*ackData)(struct IPPacket *p)); 

void SendRequest(char *filename, void (*ackData)(struct IPPacket *p));

int  PrepareRequest(char *data, size_t size, char *filename) ;
