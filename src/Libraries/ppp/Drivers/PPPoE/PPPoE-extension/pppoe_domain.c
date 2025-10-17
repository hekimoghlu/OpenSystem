/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 14, 2022.
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
#include <sys/param.h>
#include <sys/systm.h>
#include <sys/kernel.h>
#include <sys/malloc.h>
#include <sys/mbuf.h>
#include <sys/socket.h>
#include <sys/sockio.h>
#include <sys/sysctl.h>

#include <sys/syslog.h>
#include <mach/vm_types.h>
#include <mach/kmod.h>
#include <sys/socketvar.h>
#include <sys/protosw.h>
#include <sys/domain.h>
#include <kern/thread.h>
#include <kern/locks.h>
#include <net/if.h>

#include "../../../Family/if_ppplink.h"
#include "../../../Family/ppp_domain.h"
#include "PPPoE.h"
#include "pppoe_proto.h"
#include "pppoe_wan.h"
#include "pppoe_rfc.h"


/* -----------------------------------------------------------------------------
Definitions
----------------------------------------------------------------------------- */


/* -----------------------------------------------------------------------------
Forward declarations
----------------------------------------------------------------------------- */
int pppoe_domain_init(int);
int pppoe_domain_terminate(int);

/* this function has not prototype in the .h file */
struct domain *pffinddomain(int pf);

/* -----------------------------------------------------------------------------
Globals
----------------------------------------------------------------------------- */

int 		pppoe_domain_inited = 0;

extern lck_mtx_t   *ppp_domain_mutex;

/* -----------------------------------------------------------------------------
----------------------------------------------------------------------------- */
int pppoe_domain_module_start(struct kmod_info *ki, void *data)
{
    //boolean_t 	funnel_state;
    int		ret;

    //funnel_state = thread_funnel_set(network_flock, TRUE);
    ret = pppoe_domain_init(0);
    //thread_funnel_set(network_flock, funnel_state);

    return ret;
}

/* -----------------------------------------------------------------------------
----------------------------------------------------------------------------- */
int pppoe_domain_module_stop(struct kmod_info *ki, void *data)
{
    //boolean_t 	funnel_state;
    int		ret;

    //funnel_state = thread_funnel_set(network_flock, TRUE);
    ret = pppoe_domain_terminate(0);
    //thread_funnel_set(network_flock, funnel_state);

    return ret;
}

/* -----------------------------------------------------------------------------
----------------------------------------------------------------------------- */
int pppoe_domain_init(int init_arg)
{
    int 	ret = KERN_SUCCESS;
    struct domain *pppdomain;
    
    IOLog("PPPoE domain init\n");

    if (pppoe_domain_inited)
        return(KERN_SUCCESS);

    pppdomain = pffinddomain(PF_PPP);
    if (!pppdomain) {
        IOLog("PPPoE domain init : PF_PPP domain does not exist...\n");
        return(KERN_FAILURE);
    }
    	
	lck_mtx_lock(ppp_domain_mutex);
	
    ret = pppoe_rfc_init();
    if (ret) {
        IOLog("PPPoE domain init : can't init PPPoE protocol RFC, err : %d\n", ret);
        goto end;
    }
    
    ret = pppoe_add(pppdomain);
    if (ret) {
        IOLog("PPPoE domain init : can't add proto to PPPoE domain, err : %d\n", ret);
        pppoe_rfc_dispose();
        goto end;
    }

    pppoe_wan_init();

    pppoe_domain_inited = 1;

end:
	lck_mtx_unlock(ppp_domain_mutex);
    return ret;
}


/* -----------------------------------------------------------------------------
----------------------------------------------------------------------------- */
int pppoe_domain_terminate(int term_arg)
{
    int 	ret = KERN_SUCCESS;
    struct domain *pppdomain;
    
    IOLog("PPPoE domain terminate\n");

    if (!pppoe_domain_inited)
        return(KERN_SUCCESS);
		
	pppdomain = pffinddomain(PF_PPP);
    if (!pppdomain) {
        // humm.. should not happen
        IOLog("PPPoE domain terminate : PF_PPP domain does not exist...\n");
        return KERN_FAILURE;
    }

	lck_mtx_lock(ppp_domain_mutex);
	
    ret = pppoe_rfc_dispose();
    if (ret) {
        IOLog("PPPoE domain is in use and cannot terminate, err : %d\n", ret);
        goto end;
    }

    ret = pppoe_wan_dispose();
    if (ret) {
        IOLog("PPPoE domain terminate : pppoe_wan_dispose, err : %d\n", ret);
        goto end;
    }
    
    ret = pppoe_remove(pppdomain);
    if (ret) {
        IOLog("PPPoE domain terminate : can't del proto from PPPoE domain, err : %d\n", ret);
        goto end;
    }

    pppoe_domain_inited = 0;

end:
	lck_mtx_unlock(ppp_domain_mutex);
    return ret;
}
