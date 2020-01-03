SELECT cpv.value as “URL Visited”
        ,if(cpv.timestamp<>'',from_unixtime(floor(cpv.timestamp/1000)),null) as  "CPV Timestamp"
        ,cpv.name as "CPV Name"
		,c.emailAddress as "HS Email"
		,c.substring(c.emailAddress, position("@" in c.emailAddress)+1,length(c.emailAddress)) as "HS Email Domain"
		,c.profileUrl as "HS Profile URL"
		,c.contactProperty_company as "HS Company"
		,c.contactProperty_country as "HS Country"
		,c.contactProperty_hs_analytics_source_data_1 as "Original Source Type 1"
		,c.contactProperty_hs_analytics_source_data_2 as "Original Source Type 2"
		,c.contactProperty_industry as "HS Industry"
		,c.contactProperty_job_role__c as "HS Job Role"
		,c.contactProperty_lifecyclestage as "Lifecycle Stage"
		,if(c.contactProperty_hs_lifecyclestage_lead_date<>"",from_unixtime(floor(c.contactProperty_hs_lifecyclestage_lead_date/1000)),null) AS "Lifecycle Lead Date"
		,c.contactProperty_explicit_interest_analytics_cloud as "Interest in Analytics Cloud"
		,c.contactProperty_explicit_interest_commerce as "Interest in Commerce"
		,c.contactProperty_explicit_interest_dxp as "Interest in DXP"
		,c.contactProperty_salesforceaccountid as "Salesforce Account Id"
		,c.contactProperty_salesforcecontactid as "Salesforce Contact Id"
		,c.contactProperty_salesforceleadid as "Salesforce Lead Id"
		,CASE
			WHEN c.contactProperty_hs_analytics_source = "DIRECT_TRAFFIC" THEN "Direct Traffic"
			WHEN c.contactProperty_hs_analytics_source = "EMAIL_MARKETING" THEN "Email Marketing"
			WHEN c.contactProperty_hs_analytics_source = "ORGANIC_SEARCH" THEN "Organic Search"
			WHEN c.contactProperty_hs_analytics_source = "SOCIAL_MEDIA" THEN "Social Media"
			WHEN c.contactProperty_hs_analytics_source = "REFERRALS" THEN "Referrals"
			WHEN c.contactProperty_hs_analytics_source = "OFFLINE" THEN "Offline"
			WHEN c.contactProperty_hs_analytics_source = "PAID_SEARCH" THEN "Paid Search"
			WHEN c.contactProperty_hs_analytics_source = "OTHER_CAMPAIGNS" THEN "Other Campaigns"
			WHEN c.contactProperty_hs_analytics_source = "PAID_SOCIAL" THEN "Paid Social"
		ELSE "Other"
		END AS "Original Source Type"
FROM ContactPropertyVersion as cpv
INNER JOIN Contact as c
ON cpv.vid = c.vid
WHERE substring(c.emailAddress, position("@" in c.emailAddress)+1,length(c.emailAddress)) <> 'liferay.com'
AND cpv.name in ("hs_analytics_last_url", "hs_analytics_last_referrer")
