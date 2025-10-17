/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 14, 2023.
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
 * Modification History
 *
 * August 5, 2004			Allan Nathanson <ajn@apple.com>
 * - initial revision
 */

#ifndef _NET_H
#define _NET_H

#include <sys/cdefs.h>

#include <CoreFoundation/CoreFoundation.h>
#include <SystemConfiguration/SystemConfiguration.h>


typedef int (*optionHandler) (CFStringRef		key,
			      const char		*description,
			      void			*info,
			      int			argc,
			      char * const		argv[],
			      CFMutableDictionaryRef	newConfiguration);

typedef enum {
	isOther,		// use "only" handler function for processing
	isHelp,
	isChooseOne,
	isChooseMultiple,
	isBool,			// CFBoolean
	isBoolean,		// CFNumber (0 or 1)
	isNumber,		// CFNumber
	isString,		// CFString
	isStringArray		// CFArray[CFString]
} optionType;

typedef const struct {
	const CFStringRef	selection;
	const CFStringRef	*key;
	const unsigned int	flags;
} selections;
#define selectionNotAvailable	1<<0	// if you can't "choose" this selection

typedef const struct {
	const char		*option;
	const char		*description;
	optionType		type;
	const CFStringRef	*key;
	optionHandler		handler;
	void			*info;
} options, *optionsRef;


extern CFMutableArrayRef	new_interfaces;

extern CFArrayRef		interfaces;
extern CFArrayRef		services;
extern CFArrayRef		protocols;
extern CFArrayRef		sets;

extern SCNetworkInterfaceRef	net_interface;
extern SCNetworkServiceRef	net_service;
extern SCNetworkProtocolRef	net_protocol;
extern SCNetworkSetRef		net_set;

extern CFNumberRef		CFNumberRef_0;
extern CFNumberRef		CFNumberRef_1;


__BEGIN_DECLS

Boolean		_process_options(optionsRef		options,
				 int			nOptions,
				 int			argc,
				 char * const		argv[],
				 CFMutableDictionaryRef	newConfiguration);

CF_RETURNS_RETAINED
CFNumberRef	_copy_number	(const char *arg);

CFIndex		_find_option	(const char	*option,
				 optionsRef	options,
				 const int	nOptions);

CFIndex		_find_selection	(CFStringRef 	choice,
				 selections	choises[],
				 unsigned int	*flags);

void		_show_entity	(CFDictionaryRef entity, CFStringRef prefix);

void		do_net_init	(void);
void		do_net_quit	(int argc, char * const argv[]);

void		do_net_open	(int argc, char * const argv[]);
void		do_net_commit	(int argc, char * const argv[]);
void		do_net_apply	(int argc, char * const argv[]);
void		do_net_close	(int argc, char * const argv[]);

void		do_net_clean	(int argc, char * const argv[]);
void		do_net_create	(int argc, char * const argv[]);
void		do_net_disable	(int argc, char * const argv[]);
void		do_net_enable	(int argc, char * const argv[]);
void		do_net_migrate	(int argc, char * const argv[]);
void		do_net_remove	(int argc, char * const argv[]);
void		do_net_select	(int argc, char * const argv[]);
void		do_net_set	(int argc, char * const argv[]);
void		do_net_show	(int argc, char * const argv[]);
void		do_net_update	(int argc, char * const argv[]);
void		do_net_upgrade	(int argc, char * const argv[]);

void		do_net_snapshot	(int argc, char * const argv[]);

void		do_configuration(int argc, char **argv);

__END_DECLS

#endif /* !_NET_H */
