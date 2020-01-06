SELECT ContactPropertyVersion.value as “URL Visited”
        ,if(ContactPropertyVersion.timestamp<>'',from_unixtime(floor(ContactPropertyVersion.timestamp/1000)),null) as  "CPV Timestamp"
        ,ContactPropertyVersion.name as "CPV Name"
		,Contact.emailAddress as "HS Email"
		,Contact.substring(Contact.emailAddress, position("@" in Contact.emailAddress)+1,length(Contact.emailAddress)) as "HS Email Domain"
		,Contact.profileUrl as "HS Profile URL"
		,Contact.contactProperty_company as "HS Company"
		,Contact.contactProperty_country as "HS Country"
		,Contact.contactProperty_hs_analytics_source_data_1 as "Original Source Type 1"
		,Contact.contactProperty_hs_analytics_source_data_2 as "Original Source Type 2"
		,Contact.contactProperty_industry as "HS Industry"
		,Contact.contactProperty_job_role__c as "HS Job Role"
		,Contact.contactProperty_lifecyclestage as "Lifecycle Stage"
		,if(Contact.contactProperty_hs_lifecyclestage_lead_date<>"",from_unixtime(floor(Contact.contactProperty_hs_lifecyclestage_lead_date/1000)),null) AS "Lifecycle Lead Date"
		,Contact.contactProperty_explicit_interest_analytics_cloud as "Interest in Analytics Cloud"
		,Contact.contactProperty_explicit_interest_commerce as "Interest in Commerce"
		,Contact.contactProperty_explicit_interest_dxp as "Interest in DXP"
		,Contact.contactProperty_salesforceaccountid as "Salesforce Account Id"
		,Contact.contactProperty_salesforcecontactid as "Salesforce Contact Id"
		,Contact.contactProperty_salesforceleadid as "Salesforce Lead Id"
		,CASE
			WHEN Contact.contactProperty_hs_analytics_source = "DIRECT_TRAFFIC" THEN "Direct Traffic"
			WHEN Contact.contactProperty_hs_analytics_source = "EMAIL_MARKETING" THEN "Email Marketing"
			WHEN Contact.contactProperty_hs_analytics_source = "ORGANIC_SEARCH" THEN "Organic Search"
			WHEN Contact.contactProperty_hs_analytics_source = "SOCIAL_MEDIA" THEN "Social Media"
			WHEN Contact.contactProperty_hs_analytics_source = "REFERRALS" THEN "Referrals"
			WHEN Contact.contactProperty_hs_analytics_source = "OFFLINE" THEN "Offline"
			WHEN Contact.contactProperty_hs_analytics_source = "PAID_SEARCH" THEN "Paid Search"
			WHEN Contact.contactProperty_hs_analytics_source = "OTHER_CAMPAIGNS" THEN "Other Campaigns"
			WHEN Contact.contactProperty_hs_analytics_source = "PAID_SOCIAL" THEN "Paid Social"
		ELSE "Other"
		END AS "Original Source Type"
FROM ContactPropertyVersion
INNER JOIN Contact
	ON ContactPropertyVersion.vid = Contact.vid
WHERE substring(Contact.emailAddress, position("@" in Contact.emailAddress)+1,length(Contact.emailAddress)) <> 'liferay.com'
	AND ContactPropertyVersion.name in ("hs_analytics_last_url", "hs_analytics_last_referrer")
