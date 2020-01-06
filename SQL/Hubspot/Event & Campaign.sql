# Query for Emails
SELECT if(Event.created<>'',from_unixtime(floor(Event.created/1000)),null) as createddate_email
		,Event.recipient
		,Event.type
		,CampaignMember_.name
		,CampaignMember.subject
		,CampaignMember.counterBounced as Bounced
		,CampaignMember.counterclicked as Clicked
		,CampaignMember.counterDeferred as Deferred
		,CampaignMember.counterDropped as Dropped 
		,CampaignMember.counterForward as Forward
		,CampaignMember.counterProcessed as Processed
		,CampaignMember.counterOpened as Opened
		,CampaignMember.counterReplied as Replied
		,CampaignMember.counterSent as Sent
		,CampaignMember.counterUnsubscribed as Unsubscribed
		,if(CampaignMember.lastUpdatedTime<>'',from_unixtime(floor(CampaignMember.lastUpdatedTime/1000)),null) as lastUpdatedTime
FROM Event
INNER JOIN Campaign
	ON Event.emailCampaignId = CampaignMember.id
