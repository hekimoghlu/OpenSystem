/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 8, 2021.
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
 * Include necessary headers.
 */

#include "backend-private.h"


/*
 * 'main()' - Show the supplies state of a printer.
 */

int					/* O - Exit status */
main(int  argc,				/* I - Number of command-line args */
     char *argv[])			/* I - Command-line arguments */
{
  http_addrlist_t	*host;		/* Host addresses */
  int			snmp_fd;	/* SNMP socket */
  int			page_count,	/* Current page count */
			printer_state;	/* Current printer state */


  if (argc != 2)
  {
    puts("Usage: testsupplies ip-or-hostname");
    return (1);
  }

  if ((host = httpAddrGetList(argv[1], AF_UNSPEC, "9100")) == NULL)
  {
    perror(argv[1]);
    return (1);
  }

  if ((snmp_fd = _cupsSNMPOpen(host->addr.addr.sa_family)) < 0)
  {
    perror(argv[1]);
    return (1);
  }

  for (;;)
  {
    fputs("backendSNMPSupplies: ", stdout);

    if (backendSNMPSupplies(snmp_fd, &(host->addr), &page_count,
                            &printer_state))
    {
      puts("FAIL");
      return (1);
    }

    printf("backendSNMPSupplies: %s (page_count=%d, printer_state=%d)\n",
	   page_count < 0 || printer_state < CUPS_TC_other ||
	       printer_state > CUPS_TC_warmup ? "FAIL" : "PASS",
	   page_count, printer_state);

    sleep(5);
  }
}
