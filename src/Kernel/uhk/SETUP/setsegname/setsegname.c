/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 20, 2024.
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
#include <libc.h>
#include <errno.h>

#include <sys/stat.h>
#include <sys/file.h>
#include <sys/mman.h>

#include <mach-o/swap.h>

#include <stdbool.h>

/*********************************************************************
*********************************************************************/
static int
writeFile(int fd, const void * data, size_t length)
{
	int error = 0;

	if (length != (size_t)write(fd, data, length)) {
		error = -1;
	}

	if (error != 0) {
		perror("couldn't write output");
	}

	return error;
}

/*********************************************************************
*********************************************************************/
static int
readFile(const char *path, vm_offset_t * objAddr, vm_size_t * objSize)
{
	int error = -1;
	int fd;
	struct stat stat_buf;

	*objAddr = 0;
	*objSize = 0;

	do {
		if ((fd = open(path, O_RDONLY)) == -1) {
			continue;
		}

		if (fstat(fd, &stat_buf) == -1) {
			continue;
		}

		if (0 == (stat_buf.st_mode & S_IFREG)) {
			continue;
		}

		if (0 == stat_buf.st_size) {
			error = 0;
			continue;
		}

		*objSize = stat_buf.st_size;

		*objAddr = (vm_offset_t)mmap(NULL /* address */, *objSize,
		    PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE /* flags */,
		    fd, 0 /* offset */);

		if ((void *)*objAddr == MAP_FAILED) {
			*objAddr = 0;
			*objSize = 0;
			continue;
		}

		error = 0;
	} while (false);

	if (-1 != fd) {
		close(fd);
	}
	if (error) {
		fprintf(stderr, "couldn't read %s: %s\n", path, strerror(errno));
	}

	return error;
}

static void
usage(void)
{
	fprintf(stderr, "Usage: %s [-s OLDSEGNAME] [-i IGNORESEGNAME] -n NEWSEGNAME input -o output\n", getprogname());
	exit(1);
}

/*********************************************************************
*********************************************************************/
int
main(int argc, char * argv[])
{
	int                     error;
	const char            * output_name = NULL;
	const char            * input_name = NULL;
	const char            * oldseg_name = NULL;
	const char            * ignoreseg_name = NULL;
	const char            * newseg_name = NULL;
	struct mach_header    * hdr;
	struct mach_header_64 * hdr64;
	struct load_command   * cmds;
	boolean_t                   swap = false;
	uint32_t                    ncmds, cmdtype;
	uint32_t                    len;
	vm_offset_t                 input;
	vm_size_t                   input_size;
	uint32_t                    nsects = 0;
	uint32_t                * flags = NULL;
	uint32_t                    attr;
	typedef char            segname_t[16];
	segname_t             * names = NULL;
	int                     ch;


	while ((ch = getopt(argc, argv, "s:i:n:o:")) != -1) {
		switch (ch) {
		case 's':
			oldseg_name = optarg;
			break;
		case 'i':
			ignoreseg_name = optarg;
			break;
		case 'n':
			newseg_name = optarg;
			break;
		case 'o':
			output_name = optarg;
			break;
		case '?':
		default:
			usage();
		}
	}

	argc -= optind;
	argv += optind;

	if ((argc != 1) || !newseg_name || !output_name) {
		usage();
	}

	input_name = argv[0];

	error = readFile(input_name, &input, &input_size);
	if (error) {
		exit(1);
	}

	hdr = (typeof(hdr))input;
	switch (hdr->magic) {
	case MH_CIGAM:
		swap = true;
	// fall thru
	case MH_MAGIC:
		ncmds = hdr->ncmds;
		cmds  = (typeof(cmds))(hdr + 1);
		break;

	case MH_CIGAM_64:
		swap = true;
	// fall thru
	case MH_MAGIC_64:
		hdr64 = (typeof(hdr64))hdr;
		ncmds = hdr64->ncmds;
		cmds  = (typeof(cmds))(hdr64 + 1);
		break;

	default:
		fprintf(stderr, "not macho input file\n");
		exit(1);
		break;
	}

	if (swap) {
		ncmds = OSSwapInt32(ncmds);
	}
	while (ncmds--) {
		cmdtype = cmds->cmd;
		if (swap) {
			cmdtype = OSSwapInt32(cmdtype);
		}
		nsects = 0;
		len    = 0;
		if (LC_SEGMENT == cmdtype) {
			struct segment_command * segcmd;
			struct section         * sects;

			segcmd = (typeof(segcmd))cmds;
			nsects = segcmd->nsects;
			sects  = (typeof(sects))(segcmd + 1);
			names  = &sects->segname;
			flags  = &sects->flags;
			len    = sizeof(*sects);
		} else if (LC_SEGMENT_64 == cmdtype) {
			struct segment_command_64 * segcmd;
			struct section_64         * sects;

			segcmd = (typeof(segcmd))cmds;
			nsects = segcmd->nsects;
			sects  = (typeof(sects))(segcmd + 1);
			names  = &sects->segname;
			flags  = &sects->flags;
			len    = sizeof(*sects);
		}

		if (swap) {
			nsects = OSSwapInt32(nsects);
		}
		while (nsects--) {
			attr = *flags;
			if (swap) {
				attr = OSSwapInt32(attr);
			}

			if (!(S_ATTR_DEBUG & attr) && (!ignoreseg_name ||
			    0 != strncmp(ignoreseg_name, (char *)names, sizeof(*names)))) {
				if (!oldseg_name ||
				    0 == strncmp(oldseg_name, (char *)names, sizeof(*names))) {
					memset(names, 0x0, sizeof(*names));
					strncpy((char *)names, newseg_name, sizeof(*names));
				}
			}

			names = (typeof(names))(((uintptr_t) names) + len);
			flags = (typeof(flags))(((uintptr_t) flags) + len);
		}

		len = cmds->cmdsize;
		if (swap) {
			len = OSSwapInt32(len);
		}
		cmds = (typeof(cmds))(((uintptr_t) cmds) + len);
	}

	int fd = open(output_name, O_WRONLY | O_CREAT | O_TRUNC, 0755);
	if (-1 == fd) {
		error = -1;
	} else {
		error = writeFile(fd, (const void *) input, input_size);
		close(fd);
	}

	if (error) {
		fprintf(stderr, "couldn't write output: %s\n", strerror(errno));
		exit(1);
	}

	exit(0);
	return 0;
}
