/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 24, 2023.
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
#ifndef netdissect_pcap_missing_h
#define netdissect_pcap_missing_h

/*
 * Declarations of functions that might be missing from libpcap.
 */

#ifndef HAVE_PCAP_LIST_DATALINKS
extern int pcap_list_datalinks(pcap_t *, int **);
#endif

#ifndef HAVE_PCAP_DATALINK_NAME_TO_VAL
/*
 * We assume no platform has one but not the other.
 */
extern int pcap_datalink_name_to_val(const char *);
extern const char *pcap_datalink_val_to_name(int);
#endif

#ifndef HAVE_PCAP_DATALINK_VAL_TO_DESCRIPTION
extern const char *pcap_datalink_val_to_description(int);
#endif

#ifndef HAVE_PCAP_DUMP_FTELL
extern long pcap_dump_ftell(pcap_dumper_t *);
#endif

#endif /* netdissect_pcap_missing_h */
