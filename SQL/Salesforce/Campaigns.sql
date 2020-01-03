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
	,left(cm.Id_,15) as campaign_id_trimmed
FROM Campaign as cm
#JOIN on cm.campaign_id_trimmed to hubspot.Contact.recent_interaction_campaign
