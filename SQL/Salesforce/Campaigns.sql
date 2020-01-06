SELECT cm.ActualCost
		,cm.AmountAllOpportunities
		,cm.AmountWonOpportunities
		,cm.CreatedDate as CreatedDate_campaign
		,cm.EndDate as EndDate_campaign
		,cm.HierarchyActualCost
		,cm.HierarchyAmountAllOpportunities
		,cm.HierarchyAmountWonOpportunities
		,cm.HierarchyNumberOfContacts
		,cm.HierarchyNumberOfConvertedLeads
		,cm.HierarchyNumberOfOpportunities
		,cm.HierarchyNumberOfWonOpportunities
		,cm.Id_ as Id_campaign
		,cm.IsActive
		,cm.Name as Name_campaign
		,cm.NumberOfContacts
		,cm.NumberOfConvertedLeads
		,cm.NumberOfLeads
		,cm.NumberOfOpportunities
		,cm.NumberOfResponses
		,cm.NumberOfWonOpportunities
		,cm.ParentId
		,cm.StartDate as StartDate_campaign
		,cm.Status as Status_campaign
		,cm.Type_ as Type_campaign
		,cpm.CampaignId as CampaignId_cpm
		,cpm.ContactId as ContactId_cpm
		,cpm.Id_ as Id_cpm
		,cpm.LeadId as LeadId_cpm
		,left(cm.Id_,15) as campaign_id_trimmed
FROM Campaign as cm
LEFT JOIN CampaignMember as cpm
on cm.Id_ = cpm.CampaignId
# JOIN on cm.campaign_id_trimmed to hubspot.Contact.recent_interaction_campaign



